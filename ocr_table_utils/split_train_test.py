# encoding=utf-8
import os, glob, sys
import cv2
root = "/media/handsome/backupdata/hanson/ocr_table_dataset_v2"
image_root = os.path.join(root, 'Annotations')
path_list = []
for diff_anno_folder in os.listdir(image_root):
    print(diff_anno_folder)
    diff_anno_path = os.path.join(image_root, diff_anno_folder)
    for xml_pth in glob.glob(os.path.join(diff_anno_path, '*.xml')):
        path_list.append(diff_anno_folder + '/' + xml_pth.split('/')[-1].split('.')[0] + '\n')
        # img = cv2.imread(img_pth)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)

import random
random.seed(42)
test_list = random.sample(path_list, int(len(path_list) * 0.2))
train_list = list(set(path_list).difference(set(test_list)))

with open(os.path.join(root, 'trainval.txt'), 'w') as f:
    f.writelines(train_list)
with open(os.path.join(root, 'test.txt'), 'w') as f:
    f.writelines(test_list)
# print(len(test_list), len(train_list))