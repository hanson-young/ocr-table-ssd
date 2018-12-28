import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F

from ..utils import box_utils
from collections import namedtuple
import cv2
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #

def reverse_rotate(image, ori_shape, angle):
    """get reverse transform matrix!"""
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2., h / 2.)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

    M[0, 2] += ori_shape[1] / 2 - cX
    M[1, 2] += ori_shape[0] / 2 - cY

    return M


def rotate_bound(image, angle):
    """from imutils module!"""
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def rotate_map(mask, image, device):

    seg_mask = torch.squeeze(mask[0])
    seg_mask = seg_mask.cpu().detach().numpy().astype(np.float32)
    thresh_mask = (seg_mask * 255).astype(np.uint8)

    ori_image = torch.squeeze(image[0])
    ori_image = ori_image.cpu().detach().numpy().transpose((1,2,0)).astype(np.float32)


    thresh_mask[thresh_mask > 127] = 255
    thresh_mask[thresh_mask <= 127] = 0
    thresh_mask = cv2.resize(thresh_mask, (ori_image.shape[1]//2, ori_image.shape[0]//2))

    horizon_mask = cv2.Sobel(thresh_mask, cv2.CV_8UC1, 0, 1, ksize=3)

    horizon_lines = cv2.HoughLinesP(horizon_mask, 1, np.pi / 180, threshold=40, minLineLength=10, maxLineGap=10)
    draw_img = cv2.cvtColor(thresh_mask.copy() * 0, cv2.COLOR_GRAY2BGR)
    horizon_angles = []
    rotate_angle = 0
    if horizon_lines is not None:
        for l in horizon_lines:

            dx = l[0][0] - l[0][2]
            dy = l[0][1] - l[0][3]

            theta = np.arctan2(np.array([dy]), np.array([dx]))
            if theta < 0:
                theta_tmp = np.pi + theta
            else:
                theta_tmp = theta
            if (4.5 / 18 * np.pi < theta_tmp < 13.5 / 18 * np.pi):
                continue
            p1 = np.array([l[0][0], l[0][1]])
            p2 = np.array([l[0][2], l[0][3]])
            angle = theta * 180 / np.pi
            horizon_angles.append(180 + angle if angle < 0 else angle - 180)

            cv2.line(draw_img, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 0), 1)
        if len(horizon_angles) > 0:
            rotate_angle = np.array(horizon_angles, dtype=np.float32).mean()

    rotate_image = rotate_bound(ori_image, -rotate_angle)
    rotate_mask = rotate_bound(seg_mask, -rotate_angle)
    Matrix = reverse_rotate(rotate_image, ori_image.shape, rotate_angle)
    factorx = float(ori_image.shape[1]) / rotate_mask.shape[1]
    factory = float(ori_image.shape[0]) / rotate_mask.shape[0]
    factor = [factorx, factory]
    rotate_image = cv2.resize(rotate_image, (ori_image.shape[1], ori_image.shape[0]))
    rotate_mask = cv2.resize(rotate_mask, (seg_mask.shape[1], seg_mask.shape[0]))
    # cv2.imshow("ri",rotate_image)
    # cv2.imshow("om", seg_mask)
    # cv2.imshow("rm", rotate_mask)
    # cv2.waitKey(0)
    rotate_image = torch.from_numpy(rotate_image.transpose((2,0,1))).unsqueeze(0).to(device)
    rotate_mask = torch.from_numpy(rotate_mask).unsqueeze(0).unsqueeze(0).to(device)

    return rotate_mask, rotate_image, rotate_angle, Matrix, factor

class SSD(nn.Module):
    def __init__(self, num_classes: int, mask_net, base_net: nn.ModuleList, source_layer_indexes: List[int],
                 extras: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.mask_net = mask_net
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        #
        #
        x_l, x_o = self.mask_net(x)
        # sub = getattr(self.mask_net[end_layer_index], 'final')
        # for layer in sub[:path.s1]:
        #     x = layer(x)

        seg_mask = torch.sigmoid(x_l)

        if self.is_test:
            seg_mask, x_o ,angle, Matrix, factor = rotate_map(seg_mask, x_o, self.device)

        x = torch.cat([seg_mask, x_o],1)

        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes, seg_mask, angle, Matrix, factor
        else:
            return confidences, locations, seg_mask

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if
                      not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.mask_net.apply(_xavier_init_)
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance,
                                                         self.size_variance)
        return locations, labels


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
