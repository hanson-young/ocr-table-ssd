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
            '1111'
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
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            polygon = object.find('polygon')
            # VOC dataset format follows Matlab, in which indexes start from 0

            point0 = polygon.find('point0').text.split(',')
            point1 = polygon.find('point1').text.split(',')
            point2 = polygon.find('point2').text.split(',')
            point3 = polygon.find('point3').text.split(',')
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
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

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

if __name__ == "__main__":
    ocr_cropped_root = '/media/handsome/backupdata/hanson/orc_cropped'
    is_test = False
    if is_test:
        txt_file = ocr_cropped_root + "/test.txt"
    else:
        txt_file = ocr_cropped_root + "/trainval.txt"
    dataset = OcrDataset("/media/handsome/backupdata/hanson/ocr_table_dataset", is_test=is_test)

    print(len(dataset.ids))
    area_list = []
    for idx in range(len(dataset.ids)):
        image_id, annotation = dataset.get_annotation(idx)
        image = dataset._read_image(image_id)
        gt_boxes, classes, is_difficult = annotation
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) * 0
        for jdx in range(gt_boxes.shape[0]):

            box = gt_boxes[jdx, :].astype(np.int32)
            # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
            mask[box[1]: box[3],box[0]: box[2]] = mask[box[1]: box[3],box[0]: box[2]] * 0 + 1

        con_img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


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
            # xmin = max(0, xmin - 5)
            # ymin = max(0, ymin - 5)
            # xmax = min(w, xmax + 5)
            # ymax = min(h, ymax + 5)
            #
            # y1 = y + 1
            # y2 = y + h - 1
            # x1 = x + 1
            # x2 = x + w - 1
            #
            area_box = []
            area = image[y: y + h, x: x + w, :]
            for jdx in range(gt_boxes.shape[0]):
                box = gt_boxes[jdx, :].astype(np.int32)
                if box[0] >= x and box[2] <= x + w and box[1] >= y and box[3] <= y + h:
                    area_box.append([box[0] - x + 1, box[1] - y + 1, box[2] - x - 1, box[3] - y - 1])
            area_list.append([area_box, area])

            # cv2.rectangle(con_img, (x, y), (x + w, y + h), (153, 153, 0), 5)

    with open(txt_file,'w') as f:
        for area_ in area_list:

            boxes = area_[0]
            area = area_[1]
            ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 20))
            ocr_cropped_xml = os.path.join(ocr_cropped_root, 'Annotations', '01', ran_str + ".xml")
            ocr_cropped_img = os.path.join(ocr_cropped_root, 'Images', '01', ran_str + ".png")

            voc = VOCAnnotation(ran_str + ".png", area.shape[1], area.shape[0])

            for box in boxes:
                # cv2.rectangle(area, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
                voc.addBoundingBox(box[0], box[1], box[2], box[3], "1111")

            voc.save(ocr_cropped_xml)
            cv2.imwrite(ocr_cropped_img, area)
            f.write('01/'+ ran_str + '\n')
            cv2.imshow("area", area)
            cv2.waitKey(10)

        # cv2.imshow("img",image)
        # cv2.imshow("con_img", con_img)
        # cv2.imshow("mask", mask * 255)
        # cv2.waitKey(0)