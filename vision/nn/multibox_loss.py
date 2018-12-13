import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from ..utils import box_utils

class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, masks, labels, gt_locations, gt_masks):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        masks = torch.squeeze(masks)
        #
        # for idx in range(masks.size(0)):
        #     gt_mask = gt_masks[idx]
        #     seg_mask = masks[idx]
        #     gt_mask = gt_mask.cpu().numpy().astype(np.float32)
        #     seg_mask = torch.squeeze(seg_mask)
        #     seg_mask = seg_mask.cpu().detach().numpy().astype(np.float32)
        #     cv2.imshow("gt_mask", gt_mask)
        #     cv2.imshow("seg_mask", seg_mask)
        #     cv2.waitKey(0)
        mse_loss = F.mse_loss(masks, gt_masks, size_average=True)
        num_pos = gt_locations.size(0)
        return smooth_l1_loss/num_pos, classification_loss/num_pos, mse_loss
