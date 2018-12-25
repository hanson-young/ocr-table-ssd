import torch
from vision.utils.misc import str2bool, Timer
import argparse
import pathlib
import numpy as np
import logging
import sys
import random
import xml.etree.ElementTree as ET
from lxml import etree
from vision.ssd.imJnet_ssd_lite import create_imJnet_ssd_lite
from vision.ssd.imJnet_ssd_lite import create_imJnet_ssd_lite_predictor
import cv2
import string,os



parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="jnet-ssd-lite",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite, jnet-ssd-lite or vgg16-ssd.")
# model_log/jnet-ssd-lite-Epoch-210-Loss-0.6506194928113151.pth
parser.add_argument("--trained_model", default= 'model_log/jnet-ssd-lite-Epoch-20-Loss-0.7582516381234834.pth', type=str)

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--dataset", default='/home/handsome/Documents/data/test_crop', type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", default='model_log/voc-model-labels.txt' ,type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)

parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


if __name__ == '__main__':
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    timer = Timer()
    class_names = [name.strip() for name in open(args.label_file).readlines()]

    net = create_imJnet_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True)

    timer.start("Load Model")
    pretrained_dict = torch.load(args.trained_model)
    net.load_state_dict(pretrained_dict)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')

    predictor = create_imJnet_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)

    import glob
    for image_path in glob.glob(os.path.join(args.dataset, "*.jpg")):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) * 0
        max_edge = max(image.shape[0],image.shape[1])
        # detect_area = cv2.copyMakeBorder(image, 0, max_edge - image.shape[0], 0, max_edge - image.shape[1],cv2.BORDER_CONSTANT)
        # gt_mask = cv2.copyMakeBorder(mask, 0, max_edge - image.shape[0], 0, max_edge - image.shape[1],
        #                            cv2.BORDER_CONSTANT)
        detect_area = image.copy()
        gt_mask = mask.copy()
        boxes, labels, probs, masks = predictor.predict(detect_area, gt_mask)
        for i in range(boxes.size(0)):
            if probs[i] > 0.45:
                box = boxes[i, :]
                b = random.randint(0, 255)
                g = random.randint(0, 255)
                r = random.randint(0, 255)
                box[0] = int(min((box[0] + 5), image.shape[1] - 1))
                box[1] = int(min((box[1] + 5), image.shape[0] - 1))
                box[2] = int(max((box[2] - 5), box[0]))
                box[3] = int(max((box[3] - 5), box[1]))
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (b, g, r), 2)

        seg_mask = masks[0]
        seg_mask = torch.squeeze(seg_mask)
        seg_mask = seg_mask.cpu().detach().numpy().astype(np.float32)
        seg_mask = (seg_mask * 255).astype(np.uint8)
        seg_mask = cv2.applyColorMap(seg_mask, cv2.COLORMAP_JET)

        cv2.imshow('seg_mask', seg_mask)
        cv2.imshow('image', image)
        cv2.waitKey(0)
