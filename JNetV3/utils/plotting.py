import numpy as np
import cv2
import torch
import torch.nn.functional as F


def ImgTensorToNumpy(img):
    '''
    :param img: torch FloatTensor of shape (3,H,W)  RGB  or (H,W)  Heatmap
    :return:  numpy of shape (H,W,3)  RGB  or (H,W)
    '''
    if len(img.size())==3:
        return np.uint8(img.numpy().clip(0.,255.)).transpose((1,2,0))
    return img.numpy().clip(0., 255.)

def ImgNumpyToTensor(img):
    '''
    :param img: numpy of shape (H,W,3) RGB or (H,W)
    :return: torch FloatTensor of shape (3,H,W) RGB or (H,W)
    '''
    if len(img.shape)==3:
        return torch.from_numpy(img.transpose((2,0,1))).float()
    return torch.from_numpy(img).float()

def AddHeatmap(img, heatmap):
    '''
    :param img:  np.array  (H,W,3) RGB
    :param heatmap: np.array  (H,W)  OK to be not normalized
    :return:
    '''
    # normalize heatmap

    heatmap = np.maximum(heatmap, 0)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize between 0-1
    heatmap = np.uint8(heatmap * 255)  # Scale between 0-255 to visualize

    from matplotlib import pyplot as plt


    heatmap = np.repeat(heatmap[:, :, np.newaxis], 3, axis=2)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_with_heatmap = np.float32(heatmap) + np.float32(img)

    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    img_with_heatmap = np.uint8(255 * img_with_heatmap)
    return img_with_heatmap


def getPlotImg(smp_img, pred_hm, true_hm):
    H,W = smp_img.size()[-2:]

    imgs_to_plot = np.zeros((3, H, W, 3))

    img_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)).float().cuda()
    img_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)).float().cuda()

    ori_img = (smp_img.data*img_std+img_mean)*255.
    ori_img = ImgTensorToNumpy(ori_img.cpu())

    # add heatmap to origin image
    imgs_to_plot[0] = ori_img
    imgs_to_plot[1] = AddHeatmap(ori_img, ImgTensorToNumpy(true_hm.data.cpu()))
    imgs_to_plot[2] = AddHeatmap(ori_img, ImgTensorToNumpy(pred_hm.data.cpu()))

    return imgs_to_plot

