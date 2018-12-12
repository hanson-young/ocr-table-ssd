# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import cv2


class Sdata(data.Dataset):
    def __init__(self, anno_pd, transforms, dis=False):
        anno_pd.index = range(anno_pd.shape[0])
        self.image_paths = anno_pd['image_paths'].tolist()
        self.mask_paths = anno_pd['mask_paths'].tolist()
        self.mask_teacher_paths = anno_pd['mask_teacher_paths'].tolist()
        self.transforms = transforms
        self.dis = dis
        # deal with label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        img = cv2.cvtColor(cv2.imread(self.image_paths[item]), cv2.COLOR_BGR2RGB)  # [h,w,3]  RGB
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[item]), cv2.COLOR_BGR2GRAY)
        max_edge = max(img.shape[0],img.shape[1])
        img = cv2.copyMakeBorder(img, 0, max_edge - img.shape[0], 0, max_edge - img.shape[1],cv2.BORDER_CONSTANT)
        mask = cv2.copyMakeBorder(mask, 0, max_edge - mask.shape[0], 0, max_edge - mask.shape[1],cv2.BORDER_CONSTANT)

        h, w = mask.shape
        label = np.array([0]) if mask.max() == 0 else np.array([1])
        # mask[mask==2] = 1

        if self.transforms:
            img, mask, _ = self.transforms(img, mask, None)
        img = img
        mask = mask
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).float()
        label = torch.from_numpy(label).int()

        return img, mask, label


def collate_fn(batch):
    imgs = []
    masks = []
    labels = []
    for sample in batch:
        imgs.append(sample[0])
        masks.append(sample[1])
        labels.append(sample[2])

    return torch.stack(imgs, 0), \
           torch.stack(masks, 0),\
           torch.stack(labels, 0)

def collate_fn2(batch):
    imgs = []
    masks = []
    masks_teacher = []

    for sample in batch:
        imgs.append(sample[0])
        masks.append(sample[1])
        masks_teacher.append(sample[2])

    return torch.stack(imgs, 0), \
           torch.stack(masks, 0), \
           torch.stack(masks_teacher, 0)
