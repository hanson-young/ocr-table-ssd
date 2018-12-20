import numpy as np
import pathlib
import xml.etree.ElementTree as ET
import cv2
import random
from lxml import etree
import string
import os

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
        annotation_file = self.root / f"generate_bbox/{image_id}.xml"
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
                bndbox = object.find('bndbox')
                # VOC dataset format follows Matlab, in which indexes start from 0

                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text

                boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"Images/{image_id}.png"
        # print(image_file)
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


if __name__ == "__main__":
    ocr_cropped_root = '/media/handsome/backupdata/hanson/ocr_table_dataset_v2'
    is_test = False
    if is_test:
        txt_file = ocr_cropped_root + "/test.txt"
    else:
        txt_file = ocr_cropped_root + "/trainval.txt"
    dataset = OcrDataset(ocr_cropped_root, is_test=is_test)

    print(len(dataset.ids))
    area_list = []
    for idx in range(len(dataset.ids)):
        image_id, annotation = dataset.get_annotation(idx)
        image = dataset._read_image(image_id)
        gt_boxes, classes, is_difficult = annotation

        for jdx in range(gt_boxes.shape[0]):

            box = gt_boxes[jdx, :].astype(np.int32)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)

        cv2.imshow("mask_area", image)
        cv2.waitKey(0)

            # cv2.imshow("area",area)
            # cv2.imshow("mask_area", mask_area)
            #
            # cv2.waitKey(0)