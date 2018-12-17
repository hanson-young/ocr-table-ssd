import numpy as np
import cv2
import math


def precision(true_positives, predicted_positives):
    return max(min(np.array(true_positives).sum() / (np.array(predicted_positives).sum() + 0.000001), 1.0), 0.00001)


def recall(true_positives, possible_positives):
    return max(min(np.array(true_positives).sum() / (np.array(possible_positives).sum() + 0.000001), 1.0), 0.00001)

def f1_score(recalls, precisions):
    return 2. / (1. / recalls + 1. / precisions + 0.000001)

def metrics_pred(pred_b , images, masks):
    '''
    :param pred_b: shape of (bs, H, W), per pixel class
    :param gt_b:  shape of (bs, H, W), per pixel class
    :param labels:
    :return:
    '''
    bs = pred_b.shape[0]
    true_positives = []
    predicted_positives = []
    possible_positives = []
    union_areas = []

    for i in range(bs):
        heatmap = pred_b[i].copy()
        pred_b[i][pred_b[i] > 0.5] = 1
        pred_b[i][pred_b[i] <= 0.5] = 0

        masks[i][masks[i] > 0.5] = 1
        masks[i][masks[i] <= 0.5] = 0

        predicted_positive = pred_b[i].astype(np.int32).sum()
        possible_positive = masks[i].astype(np.int32).sum()
        true_positive = np.logical_and(pred_b[i], masks[i]).astype(np.int32).sum()

        union_area = np.logical_or(pred_b[i], masks[i]).sum()

        im_color = cv2.applyColorMap((pred_b[i] * 250).astype(np.uint8), cv2.COLORMAP_JET)
        # cv2.imshow("pred",im_color)
        # cv2.imshow("mask", masks[i])
        # cv2.imshow("images", cv2.cvtColor(np.transpose(images[i], (1, 2, 0)),cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)

        true_positives.append(true_positive)
        predicted_positives.append(predicted_positive)
        possible_positives.append(possible_positive)
        union_areas.append(union_area)

    return true_positives, predicted_positives, possible_positives, union_areas


