
import cv2
import numpy as np

def rotate_bound(image, angle):
    """from imutils module!"""
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH)), M

pts = np.array([[10, 10, 1], [50, 100, 1], [30, 200, 1], [300, 40, 1]])
ori_image = cv2.imread("/home/handsome/Documents/code/pytorch-ocr/ocr-table-ssd/eval_results/example_img.png")
for pt in pts:
    cv2.circle(ori_image, (pt[0], pt[1]),1,(0,255,255), 3)
rotate_image, M = rotate_bound(ori_image, -10)
dst_ps = np.dot(M, pts.T).T
for pt in dst_ps:
    cv2.circle(rotate_image, (int(pt[0]), int(pt[1])),1,(0,255,255), 3)
cv2.imshow("img", ori_image)
cv2.imshow("rot", rotate_image)
cv2.waitKey(0)
