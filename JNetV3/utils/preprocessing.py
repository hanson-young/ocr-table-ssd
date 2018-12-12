from copy import copy
import torch
import numpy as np
from torchvision import datasets, transforms
from JNetV3.Sdata.Saug import *
import torch.utils.data as torchdata
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from JNetV3.Sdata.Sdataset import *
import glob

class trainAug(object):
    def __init__(self):
        self.augment = Compose([
            RandomResizedCrop(size=(256, 256)),
            # ResizeImg(size=(224, 224)),
            RandomHflip(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, *args):
        return self.augment(*args)

class valAug(object):
    def __init__(self):
        self.augment = Compose([
            ResizeImg(size=(256, 256)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, *args):
        return self.augment(*args)


def get_train_val(img_root, test_size=0.1, random_state=42, dis=None):
    img_paths = sorted(list(glob.glob(os.path.join(img_root, "image", "*.jpg"))))
    mask_paths = [img_root + '/mask/' + p.split('/')[-1] for p in img_paths]
    if dis is None:
        mask_teacher_paths = mask_paths
    else:
        mask_teacher_paths = ['/'.join(p.split('/')[:-2]) + '/%s/'%dis + p.split('/')[-1].replace('.jpg', '.npy') for p in
                      img_paths]
    anno = pd.DataFrame({'image_paths': img_paths,
                         'mask_paths': mask_paths,
                         'mask_teacher_paths':mask_teacher_paths,
                         })

    if test_size == 1.0:
        return None, anno
    elif test_size == 0.0:
        return anno, None

    train_pd, val_pd = train_test_split(anno, test_size=test_size, random_state=random_state)

    train_pd.index = range(train_pd.shape[0])
    val_pd.index = range(val_pd.shape[0])

    return train_pd, val_pd


def gen_dataloader(train_pd, val_pd, trainAug, valAug, train_bs =8, val_bs=4,
                   train_shuffle=True, val_shuffle=False, dis=None):

    data_set = {}
    data_loader = {}

    data_set['val'] = Sdata(val_pd, valAug)
    data_loader['val'] = torchdata.DataLoader(data_set['val'], val_bs, num_workers=4,
                                              shuffle=val_shuffle, pin_memory=True, collate_fn=collate_fn)

    if dis is None:
        data_set['train'] = Sdata(train_pd, trainAug)
        data_loader['train'] = torchdata.DataLoader(data_set['train'], train_bs, num_workers=4,
                                                    shuffle=train_shuffle, pin_memory=True,collate_fn=collate_fn)

    else:
        data_set['train'] = Sdata(train_pd, trainAug, dis=True)
        data_loader['train'] = torchdata.DataLoader(data_set['train'], train_bs, num_workers=4,
                                                    shuffle=train_shuffle, pin_memory=True, collate_fn=collate_fn2)

    return data_set,data_loader
