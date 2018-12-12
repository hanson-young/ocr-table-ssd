import cv2
import numpy as np


gun = cv2.imread("gun.jpg")

gray = cv2.cvtColor(gun,cv2.COLOR_BGR2GRAY)
gray = gray[:, ::-1]
cv2.imshow("gray",gray)
cv2.waitKey(0)