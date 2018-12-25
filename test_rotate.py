import cv2
import numpy as np


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2., h / 2.)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = (h * sin) + (w * cos)
    nH = (h * cos) + (w * sin)

    # adjust the rotation matrix to take into account translation
    M[0, 2] += ((nW / 2) - cX)
    M[1, 2] += ((nH / 2) - cY)

    # perform the actual rotation and return the image
    # M = M.astype(np.int32)
    return cv2.warpAffine(image, M, (int(nW), int(nH)))

def reverse_rotate(image, ori_shape, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2., h / 2.)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = (h * sin) + (w * cos)
    nH = (h * cos) + (w * sin)

    # adjust the rotation matrix to take into account translation
    M[0, 2] += ori_shape[1] / 2 - cX
    M[1, 2] += ori_shape[0] / 2 - cY

    # perform the actual rotation and return the image
    # M = M.astype(np.int32)
    pts = np.array([[w / 3.,h / 3.,1],[w / 2.,h / 2.,1]])
    print(pts.T.shape, M.shape)
    dsts = np.dot(M ,pts.T).T
    print(dsts)
    final = cv2.warpAffine(image, M, (ori_shape[1], ori_shape[0]))
    cv2.circle(final, (int(dsts[0, 0]),int(dsts[0, 1])),2,(255,0,0),2)
    cv2.circle(final, (int(dsts[1, 0]), int(dsts[1, 1])), 2, (255, 0, 0), 2)


    cv2.circle(image, (int(pts[0, 0]),int(pts[0, 1])),2,(255,0,0),2)
    cv2.circle(image, (int(pts[1, 0]), int(pts[1, 1])), 2, (255, 0, 0), 2)

    cv2.imshow("iii", image)
    return final
# def in_rotate(image, ori, angle):
#     # grab the dimensions of the image and then determine the
#     # center
#     (h, w) = ori.shape[:2]
#     (cX, cY) = (w / 2., h / 2.)
#
#     # grab the rotation matrix (applying the negative of the
#     # angle to rotate clockwise), then grab the sine and cosine
#     # (i.e., the rotation components of the matrix)
#
#     M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
#     cos = np.abs(M[0, 0])
#     sin = np.abs(M[0, 1])
#
#     # compute the new bounding dimensions of the image
#     nW = (h * sin) + (w * cos)
#     nH = (h * cos) + (w * sin)
#     nH, nW = image.shape[:2]
#     # adjust the rotation matrix to take into account translation
#     M[0, 2] = M[0, 2] + ((nW / 2) - cX)
#     M[1, 2] = M[1, 2] + ((nH / 2) - cY)
#
#     # perform the actual rotation and return the image
#     return cv2.warpAffine(image, M, (ori.shape[1], ori.shape[0]))


import glob
import os
root = '/media/handsome/backupdata/hanson/ocr_table_dataset_v2/Images/01'
for path in glob.glob(os.path.join(root, '*.png')):
    ori_img = cv2.imread(path)
    # ori_img = cv2.resize(ori_img,(ori_img.shape[1] * 2, ori_img.shape[0] * 2))
    angle = -10
    rotate_image = rotate_bound(ori_img, -angle)
    in_rotate_image = reverse_rotate(rotate_image, ori_img.shape, angle)
    # cv2.imshow("rotate_img", rotate_image)
    cv2.imshow("in_rotate_img", in_rotate_image)
    cv2.imshow("ori_img", ori_img)
    cv2.waitKey(0)

