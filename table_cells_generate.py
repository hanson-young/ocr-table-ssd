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
parser.add_argument("--trained_model", default= 'model_log/jnet-ssd-lite-Epoch-210-Loss-0.6506194928113151.pth', type=str)

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--dataset", default='/media/handsome/backupdata/hanson/ocr_table_dataset_v2', type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", default='model_log/voc-model-labels.txt' ,type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)

parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


class OcrDataset:

    def __init__(self, root, is_test = None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)

        if is_test:
            image_sets_file = self.root / "test.txt"
        else:
            image_sets_file = self.root / "trainval.txt"
        self.ids = OcrDataset._read_image_ids(image_sets_file)

        self.class_names = ('BACKGROUND',
            'qqq'
        )
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}


    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        et = ET.parse(annotation_file)
        objects = et.findall("object")
        size = et.findall("size")
        w = int(size[0].find('width').text)
        h = int(size[0].find('height').text)
        boxes = []
        polyes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            if class_name == 'qqq':
                polygon = object.find('polygon')
                # VOC dataset format follows Matlab, in which indexes start from 0

                point0 = polygon.find('point0').text.split(',')
                point1 = polygon.find('point1').text.split(',')
                point2 = polygon.find('point2').text.split(',')
                point3 = polygon.find('point3').text.split(',')

                point0[0] = int(point0[0])
                point0[1] = int(point0[1])
                point1[0] = int(point1[0])
                point1[1] = int(point1[1])
                point2[0] = int(point2[0])
                point2[1] = int(point2[1])
                point3[0] = int(point3[0])
                point3[1] = int(point3[1])

                polyes.append([point0, point1, point2, point3])

                xmin = min([int(point0[0]),int(point1[0]),int(point2[0]),int(point3[0])])
                ymin = min([int(point0[1]),int(point1[1]),int(point2[1]),int(point3[1])])
                xmax =max([int(point0[0]),int(point1[0]),int(point2[0]),int(point3[0])])
                ymax =max([int(point0[1]),int(point1[1]),int(point2[1]),int(point3[1])])

                xmin = max(0, xmin - 5)
                ymin = max(0, ymin - 5)
                xmax = min(w, xmax + 5)
                ymax = min(h, ymax + 5)
                boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(polyes, dtype=np.int32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8), w, h)

    def _read_image(self, image_id):
        image_file = self.root / f"Images/{image_id}.png"
        # print(image_file)
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class VOCAnnotation(object):
    def __init__(self, imageFileName, width, height):
        annotation = etree.Element("annotation")
        self.__newTextElement(annotation, "folder", "VOC2007")
        self.__newTextElement(annotation, "filename", imageFileName)

        source = self.__newElement(annotation, "source")
        self.__newTextElement(source, "database", "OCR")

        size = self.__newElement(annotation, "size")
        self.__newIntElement(size, "width", width)
        self.__newIntElement(size, "height", height)
        self.__newIntElement(size, "depth", 3)

        self.__newTextElement(annotation, "segmented", "0")

        self._annotation = annotation

    def __newElement(self, parent, name):
        node = etree.SubElement(parent, name)
        return node

    def __newTextElement(self, parent, name, text):
        node = self.__newElement(parent, name)
        node.text = text

    def __newIntElement(self, parent, name, num):
        node = self.__newElement(parent, name)
        node.text = "%d" % num

    def addBoundingBox(self, xmin, ymin, xmax, ymax, name):
        object = self.__newElement(self._annotation, "object")

        self.__newTextElement(object, "name", name)
        self.__newTextElement(object, "pose", "Unspecified")
        self.__newTextElement(object, "truncated", "0")
        self.__newTextElement(object, "difficult", "0")

        bndbox = self.__newElement(object, "bndbox")
        self.__newIntElement(bndbox, "xmin", xmin)
        self.__newIntElement(bndbox, "ymin", ymin)
        self.__newIntElement(bndbox, "xmax", xmax)
        self.__newIntElement(bndbox, "ymax", ymax)

    def save(self, saveFileName):
        tree = etree.ElementTree(self._annotation)
        tree.write(saveFileName, pretty_print=True)


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

    dataset = OcrDataset(args.dataset, is_test=True)
    print(len(dataset.ids))
    detect_list = []
    for idx in range(len(dataset.ids)):
        image_id, annotation = dataset.get_annotation(idx)
        image = dataset._read_image(image_id)
        gt_boxes, polyes, classes, is_difficult, width, height = annotation
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) * 0
        total_gt_mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) * 0
        cv2.polylines(total_gt_mask, polyes, 1, 255, 2)
        for jdx in range(gt_boxes.shape[0]):

            box = gt_boxes[jdx, :].astype(np.int32)
            # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
            mask[box[1]: box[3],box[0]: box[2]] = mask[box[1]: box[3],box[0]: box[2]] * 0 + 1

        con_img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detect_boxes = []
        for jdx in range(0, len(contours)):
            x, y, w, h = cv2.boundingRect(contours[jdx])
            detla_x = random.randint(4, 8)
            detla_y = random.randint(4, 8)
            detla_w = random.randint(8, 16)
            detla_h = random.randint(8, 16)
            x = max(0, x - detla_x)
            y = max(0, y - detla_y)
            w = min(x + w + detla_w, image.shape[1]) - x
            h = min(y + h + detla_h, image.shape[0]) - y

            detect_box = []
            detect_area = image[y: y + h, x: x + w, :]
            gt_mask = total_gt_mask[y: y + h, x: x + w]
            max_edge = max(detect_area.shape[0],detect_area.shape[1])
            factor_xy = float(max_edge)/ 768
            factor_xy = 1
            detect_area = cv2.copyMakeBorder(detect_area, 0, max_edge - detect_area.shape[0], 0, max_edge - detect_area.shape[1],cv2.BORDER_CONSTANT)
            gt_mask = cv2.copyMakeBorder(gt_mask, 0, max_edge - detect_area.shape[0], 0, max_edge - detect_area.shape[1],cv2.BORDER_CONSTANT)

            boxes, labels, probs, masks = predictor.predict(detect_area, gt_mask)
            for i in range(boxes.size(0)):
                if probs[i] > 0.45:
                    box = boxes[i, :]
                    b = random.randint(0, 255)
                    g = random.randint(0, 255)
                    r = random.randint(0, 255)
                    box[0] = int(min(x + factor_xy * (box[0] + 5), image.shape[1] - 1))
                    box[1] = int(min(y + factor_xy * (box[1] + 5), image.shape[0] - 1))
                    box[2] = int(max(x + factor_xy * (box[2] - 5), box[0]))
                    box[3] = int(max(y + factor_xy * (box[3] - 5), box[1]))
                    # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (b, g, r), 2)
                    detect_boxes.append(box)
            # cv2.imshow('image',image)
            # cv2.waitKey(1)
        detect_list.append([image_id, detect_boxes, width, height])

    gen_bbox_path = os.path.join(args.dataset,'generate_bbox')
    for idx, item in enumerate(detect_list):
        image_id = item[0]
        detect_boxes = item[1]
        width = item[2]
        height = item[3]

        voc = VOCAnnotation(image_id, width, height)

        for box in detect_boxes:
            # cv2.rectangle(area, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
            voc.addBoundingBox(box[0], box[1], box[2], box[3], "qqq")

        xml_path = os.path.join(gen_bbox_path, image_id + ".xml")
        if not os.path.exists(os.path.dirname(xml_path)):
            os.makedirs(os.path.dirname(xml_path))
        voc.save(xml_path)
