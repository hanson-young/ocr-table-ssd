import pathlib
import numpy as np
import cv2
import sys
import random
import string
import argparse
from vision.datasets.voc_dataset import VOCDataset
from vision.utils import box_utils, measurements


parse = argparse.ArgumentParser(description="Grad method for table ceils separation")
parse.add_argument("--dataset", default='/media/handsome/backupdata/hanson/orc_cropped',type=str)
parse.add_argument("--label_file",default="model_log/voc-model-labels.txt",type=str)
parse.add_argument("--prediction_file",default="eval_results/det_test_1111.txt",type=str)
parse.add_argument("--dataset_type", default="voc",type=str)

args = parse.parse_args()


if __name__ == "__main__":
    # root_path = pathlib.Path(args.root_path)
    # root_path.mkdir(exist_ok=True)
    class_names = [name.strip() for name in open(args.label_file).readlines()]
    dataset = []
    if args.dataset_type == "voc":
        dataset = VOCDataset(args.dataset, is_test=True)

    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        orig_image = dataset._read_image(image_id)
        mask = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY) * 0
        for jdx in range(gt_boxes.shape[0]):

            box = gt_boxes[jdx, :].astype(np.int32)
            # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
            mask[box[1]: box[3],box[0]: box[2]] = mask[box[1]: box[3],box[0]: box[2]] * 0 + 1
            box[0] = min(box[0] + 5, orig_image.shape[1] - 1)
            box[1] = min(box[1] + 5, orig_image.shape[0] - 1)
            box[2] = max(box[2] - 5, box[0])
            box[3] = max(box[3] - 5, box[1])
            cv2.circle(orig_image,(box[0],box[1]), 2, (255,0,0))
            cv2.circle(orig_image, (box[2], box[3]), 2, (255, 0, 0))
        con_img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # for jdx in range(0, len(contours)):
        #     x, y, w, h = cv2.boundingRect(contours[jdx])
        epsilon = 0.01 * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)
        cv2.polylines(orig_image, [approx], True, (0,255,0),2)

        # for idx, box in enumerate(gt_boxes):
        #     b = random.randint(0, 255)
        #     g = random.randint(0, 255)
        #     r = random.randint(0, 255)
        #     gt_boxes[idx][0] = min(box[0] + 5, orig_image.shape[1] - 1)
        #     gt_boxes[idx][1] = min(box[1] + 5, orig_image.shape[0] - 1)
        #     gt_boxes[idx][2] = max(box[2] - 5, box[0])
        #     gt_boxes[idx][3] = max(box[3] - 5, box[1])
        #     box = gt_boxes[idx]
        #     cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (b, g, r), 2)

        cv2.imshow("img", orig_image)
        cv2.imshow("mask", mask * 250)
        cv2.waitKey(0)


    with open(args.prediction_file) as f:
        image_ids = []
        pred_boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            # image_ids.append(t[0])
            # scores.append(float(t[1]))
            # bbox = [int(float(v)) for v in t[2:]]
            # pred_boxes.append(bbox)
