import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from .utils import BBoxTransform, ClipBoxes
from ..utils.utils import postprocess, invert_affine, display


def calc_iou(a, b):
    # a(anchor) [A, (y1, x1, y2, x2)]
    # b(gt, coco-style) [GT, (x1, y1, x2, y2)]

    # (A,)
    ay1, ax1, ay2, ax2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    aw, ah = ax2 - ax1, ay2 - ay1
    bw, bh = bx2 - bx1, by2 - by1

    # unsqueeze is key
    # (A, 1) _op_ (GT,) gets broadcasted to (A, 1) _op_ (A, GT)
    # https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
    # (A, GT)
    iw = torch.min(ax2.unsqueeze(1), bx2) - torch.max(ax1.unsqueeze(1), bx1)
    ih = torch.min(ay2.unsqueeze(1), by2) - torch.max(ay1.unsqueeze(1), by1)

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    # (A, GT)
    intersection = iw * ih

    # (A, 1)
    aarea = torch.unsqueeze(aw * ah, 1)
    # (A,)
    barea = bw * bh

    # (A, GT)
    union = (aarea + barea) - intersection
    union = torch.clamp(union, min=1e-8)

    IoU = intersection / union
    return IoU


class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2., accept_IoU_above=.5, reject_IoU_below=.4, clf_loss_weight=1., reg_loss_weight=50.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.accept_IoU_above = accept_IoU_above
        self.reject_IoU_below = reject_IoU_below
        self.clf_loss_weight = clf_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.reg_smooth_l1_loss_scaling_factor = 9.


    def forward(self, inputs, **kwargs):
        classifications, regressions, anchors, annotations = inputs

        anchors = anchors.squeeze()
        a_widths  = anchors[:, 3] - anchors[:, 1]
        a_heights = anchors[:, 2] - anchors[:, 0]
        a_ctr_x   = anchors[:, 1] + 0.5 * a_widths
        a_ctr_y   = anchors[:, 0] + 0.5 * a_heights
        anchors_cxcywh = (a_ctr_x, a_ctr_y, a_widths, a_heights)

        batch_size, num_anchors, num_classes = classifications.shape

        classifications = torch.clamp(classifications, 1e-4, 1.0 - 1e-4)

        classification_losses = torch.zeros((batch_size,), device=anchors.device, dtype=anchors.dtype)
        regression_losses     = torch.zeros((batch_size,), device=anchors.device, dtype=anchors.dtype)

        for j, (clf, reg, ann) in enumerate(zip(classifications, regressions, annotations)):
            clf_loss, reg_loss = self.single_image_loss(
                                    anchors, anchors_cxcywh, clf, reg, ann, 
                                    alpha=self.alpha, gamma=self.gamma,
                                    accept_IoU_above=self.accept_IoU_above, reject_IoU_below=self.reject_IoU_below
                                )
            classification_losses[j] += clf_loss
            regression_losses[j]     += reg_loss

        clf_loss = self.clf_loss_weight * classification_losses.mean()
        reg_loss = self.reg_loss_weight * regression_losses.mean()
        return clf_loss, reg_loss


    def single_image_loss(self, anchors, anchors_cxcywh, clf, reg, ann, alpha, gamma, accept_IoU_above, reject_IoU_below):
            # Let A = num_anchors, C = num_classes
            # (A, C)
            clf = torch.clamp(clf, 1e-4, 1.0 - 1e-4)
            # (GT, 5)
            gt = ann[ann[:, 4] != -1]

            if len(gt) == 0:
                bce = -(torch.log(1. - clf))
                clf_loss = (1 - alpha) * torch.pow(clf, gamma) * bce
                return clf_loss.sum()

            # (GT,)
            gt_labels = gt[:, -1].long() - 1 # make zero-indexed
            # (GT, 4)
            gt_boxes  = gt[:, : -1]

            match_info = self.match_anchors_to_ground_truth(
                            anchors, gt_labels, gt_boxes, 
                            accept_IoU_above=accept_IoU_above, reject_IoU_below=reject_IoU_below
                        )
            _, _, matched_anchors_mask, unmatched_anchors_mask, matched_gt_labels, matched_gt_boxes = match_info
            num_matched_anchors = matched_anchors_mask.sum()

            # (A, C)
            clf_targets = self.make_clf_targets(
                            clf, 
                            matched_anchors_mask, unmatched_anchors_mask, 
                            matched_gt_labels
                        )

            # compute the loss for classification
            clf_loss = self.focal_clf_loss(clf, clf_targets, num_matched_anchors, alpha, gamma)

            if num_matched_anchors == 0:
                reg_loss = 0.
            else:
                reg_targets = self.make_regression_targets(anchors_cxcywh, matched_anchors_mask, matched_gt_boxes)
                reg_loss = self.regression_loss(reg[matched_anchors_mask], reg_targets)

            return clf_loss, reg_loss


    def focal_clf_loss(self, clf, clf_targets, num_matched_anchors, alpha, gamma):
        # (num_matched_anchors, C)
        clf_losses = -torch.where(
            clf_targets == 1, 
            alpha       * torch.pow(1 - clf, gamma) * torch.log(    clf), 
            (1 - alpha) * torch.pow(    clf, gamma) * torch.log(1 - clf)
        )
        # no loss for anchors with .4 < IoU < .5
        clf_losses[clf_targets == -1] = 0
        clf_loss = clf_losses.sum() / max(1, num_matched_anchors)

        return clf_loss

    def regression_loss(self, reg, reg_targets):
        r = self.reg_smooth_l1_loss_scaling_factor
        reg_loss = F.smooth_l1_loss(reg * r, reg_targets * r)
        return reg_loss

    def match_anchors_to_ground_truth(self, anchors, gt_labels, gt_boxes, accept_IoU_above=.5, reject_IoU_below=.4):
        # (A, GT)
        IoU = calc_iou(anchors, gt_boxes)
        # (A,), (A,)
        IoU_max, IoU_argmax = torch.max(IoU, dim=-1)

        # (A,)
        matched_anchors_mask = IoU_max > accept_IoU_above
        # (A,)
        unmatched_anchors_mask = IoU_max < reject_IoU_below

        # (num_matched_anchors, )
        matched_gt_indices = IoU_argmax[matched_anchors_mask]
        # (num_matched_anchors, )
        matched_gt_labels  = gt_labels[ matched_gt_indices ]
        # (num_matched_anchors, 4)
        matched_gt_boxes   = gt_boxes [ matched_gt_indices ]

        return IoU_max, IoU_argmax, \
                matched_anchors_mask, unmatched_anchors_mask, \
                matched_gt_labels, matched_gt_boxes

    def make_regression_targets(self, anchors_info, matched_mask, matched_gt_boxes):

        # (num_matched_anchors, )
        a_ctr_x, a_ctr_y, a_widths, a_heights = [a[matched_mask] for a in anchors_info]

        # (num_matched_anchors, 4)
        gt_widths  = matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0]
        gt_heights = matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1]
        gt_ctr_x = matched_gt_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = matched_gt_boxes[:, 1] + 0.5 * gt_heights

        # efficientdet style
        gt_widths  = torch.clamp(gt_widths, min=1)
        gt_heights = torch.clamp(gt_heights, min=1)

        # (A, )
        targets_dx = (gt_ctr_x - a_ctr_x) / a_widths
        targets_dy = (gt_ctr_y - a_ctr_y) / a_heights
        targets_dw = torch.log(gt_widths  / a_widths)
        targets_dh = torch.log(gt_heights / a_heights)

        # (A, 4)
        reg_targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw), dim=-1)
        return reg_targets

    def make_clf_targets(self, clf, matched_anchors_mask, unmatched_anchors_mask, matched_gt_labels):

        # (A, C)
        clf_targets = torch.empty_like(clf).fill_(-1)

        clf_targets[unmatched_anchors_mask, :] = 0.

        # one-hot encode classes
        num_classes = clf.shape[-1]
        if num_classes == 1:
            clf_targets[matched_anchors_mask, 0] = 1.
        else:
            clf_targets[matched_anchors_mask, :] = 0.
            clf_targets[matched_anchors_mask, matched_gt_labels] = 1.

        return clf_targets
