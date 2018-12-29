import torch

from vision.datasets.voc_dataset import VOCDataset

from vision.utils.misc import str2bool, Timer
import argparse
import pathlib
import numpy as np

import random

from vision.ssd.imJnet_ssd_lite import create_imJnet_ssd_lite
from vision.ssd.imJnet_ssd_lite import create_imJnet_ssd_lite_predictor
import cv2
import string
import imutils
parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
# model_log/jnet-ssd-lite-Epoch-210-Loss-0.6506194928113151.pth
parser.add_argument("--trained_model", default= 'model_log/jnet-ssd-lite-Epoch-1080-Loss-0.6063801158558239.pth', type=str)

parser.add_argument("--dataset", default='/media/handsome/backupdata/hanson/ocr_table_dataset_v2/Cropped_v1', type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", default='model_log/voc-model-labels.txt' ,type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
# DEVICE = torch.device("cpu")

def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases

if __name__ == '__main__':
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    timer = Timer()
    class_names = [name.strip() for name in open(args.label_file).readlines()]
    dataset = VOCDataset(args.dataset, is_test=True)

    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)

    net = create_imJnet_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True)

    import collections
    timer.start("Load Model")
    # net.load(args.trained_model)
    pretrained_dict = torch.load(args.trained_model)
    # new_state_dict = collections.OrderedDict()
    # for k, v in pretrained_dict.items():
    #     print(k)
    #     name = k[7:]
    #     new_state_dict[name] = v
    net.load_state_dict(pretrained_dict)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')

    predictor = create_imJnet_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)

    for i in range(len(dataset)):
        print("process image", i)
        timer.start("Load Image")
        src_image, gt_mask = dataset.get_image(i)
        # image = imutils.resize(image,image.shape[1], image.shape[0] * 2)
        image = cv2.resize(src_image,(src_image.shape[1], src_image.shape[0] * 2))
        max_edge = max(image.shape[0], image.shape[1])
        image = cv2.copyMakeBorder(image, 0, max_edge - image.shape[0], 0, max_edge - image.shape[1],cv2.BORDER_CONSTANT)

        print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        timer.start("Predict")
        boxes, labels, probs, masks, angle, Matrix, factor= predictor.predict(image, gt_mask)
        print("Prediction: {:4f} seconds.".format(timer.end("Predict")))

        orig_image = image.copy()
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation

        seg_mask = masks[0]
        seg_mask = torch.squeeze(seg_mask)
        seg_mask = seg_mask.cpu().detach().numpy().astype(np.float32)

        ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 20))
        ocr_cropped_bbox = 'eval_results/bbox/'+ ran_str + ".png"
        ocr_cropped_heatmap = 'eval_results/heatmap/' + ran_str + ".png"
        print(factor)
        final_pts = []
        for i in range(boxes.size(0)):
            if probs[i] > 0.4:
                box = boxes[i, :]

                x1 = min(box[0] + 5, 768-1) / factor[0]
                y1 = min(box[1] + 5, 768-1) / factor[1]
                x2 = max(box[2] - 5, box[0]) / factor[0]
                y2 = max(box[3] - 5, box[1]) / factor[1]
                pts = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]])
                # print(pts.T.shape, M.shape)
                dst_ps = np.dot(Matrix, pts.T).T
                for p in range(dst_ps.shape[0]):
                    dst_ps[p, 0] = int(dst_ps[p, 0] * (image.shape[1] / 768.0))
                    dst_ps[p, 1] = int(dst_ps[p, 1] * (image.shape[1] / 768.0) / 2.0)

                final_pts.append(dst_ps.astype(np.int32))

        for dps in final_pts:
            b = random.randint(0, 255)
            g = random.randint(0, 255)
            r = random.randint(0, 255)
            cv2.line(src_image, (dps[0,0],dps[0,1]), (dps[2,0],dps[2,1]), (b, g, r), 2)
            # for p in range(dps.shape[0]):
                # cv2.circle(src_image, (dps[p,0], dps[p,1]), 2, (b, g, r), 2)
        cv2.imshow('src_image',src_image)
        cv2.imshow('seg_mask', seg_mask)
        cv2.waitKey(0)
