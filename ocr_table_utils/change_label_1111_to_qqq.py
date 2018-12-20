import numpy as np
import pathlib
import xml.etree.ElementTree as ET
import cv2
import random
from lxml import etree
import string
import glob
import os

def _get_annotation(annotation_file):
    et = ET.parse(annotation_file)
    objects = et.findall("object")
    size = et.findall("size")
    w = int(size[0].find('width').text)
    h = int(size[0].find('height').text)

    polyes = []
    for object in objects:
        class_name = object.find('name').text.lower().strip()
        if class_name == '1111':
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

    return polyes, w, h




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
    def addPoints(self, pt0, pt1, pt2, pt3, name):
        object = self.__newElement(self._annotation, "object")

        self.__newTextElement(object, "name", name)
        self.__newTextElement(object, "pose", "Unspecified")
        self.__newTextElement(object, "truncated", "0")
        self.__newTextElement(object, "difficult", "0")

        polygon = self.__newElement(object, "polygon")
        self.__newTextElement(polygon, "point0", str(int(pt0[0])) + ',' + str(int(pt0[1])))
        self.__newTextElement(polygon, "point1", str(int(pt1[0])) + ',' + str(int(pt1[1])))
        self.__newTextElement(polygon, "point2", str(int(pt2[0])) + ',' + str(int(pt2[1])))
        self.__newTextElement(polygon, "point3", str(int(pt3[0])) + ',' + str(int(pt3[1])))
    def save(self, saveFileName):
        tree = etree.ElementTree(self._annotation)
        tree.write(saveFileName, pretty_print=True)

if __name__ == "__main__":
    ocr_cropped_root = os.path.join('/media/handsome/backupdata/hanson/1111', "Annotations")
    for annos_folder in os.listdir(ocr_cropped_root):
        print(annos_folder)
        annos_path = os.path.join(ocr_cropped_root, annos_folder)
        for annos_file in glob.glob(os.path.join(annos_path, "*.xml")):

            polyes, w, h = _get_annotation(annos_file)

            voc = VOCAnnotation(annos_file.split('/')[-1].replace('.xml', '.png'), w, h)
            for poly in polyes:

                # cv2.rectangle(area, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
                voc.addPoints(poly[0], poly[1], poly[2], poly[3], "qqq")
            voc.save(annos_file)
    # with open(txt_file,'w') as f:
    #     for area_ in area_list:
    #
    #         boxes = area_[0]
    #         area = area_[1]
    #         mask_area = area_[2]
    #         ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 20))
    #         ocr_cropped_xml = os.path.join(ocr_cropped_root, 'Annotations', '01', ran_str + ".xml")
    #         ocr_cropped_img = os.path.join(ocr_cropped_root, 'Images', '01', ran_str + ".png")
    #         ocr_cropped_mask = os.path.join(ocr_cropped_root, 'Segmentations', '01', ran_str + ".png")
    #         voc = VOCAnnotation(ran_str + ".png", area.shape[1], area.shape[0])
    #
    #         for box in boxes:
    #             # cv2.rectangle(area, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
    #             voc.addBoundingBox(box[0], box[1], box[2], box[3], "1111")
    #
    #         voc.save(ocr_cropped_xml)
    #         cv2.imwrite(ocr_cropped_img, area)
    #         cv2.imwrite(ocr_cropped_mask, mask_area)
    #         f.write('01/'+ ran_str + '\n')
    #         cv2.imshow("area", area)
    #         cv2.imshow("mask_area", mask_area)
    #         cv2.waitKey(10)
    #
    #         # cv2.imshow("area",area)
    #         # cv2.imshow("mask_area", mask_area)
    #         #
    #         # cv2.waitKey(0)