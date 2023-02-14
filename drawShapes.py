import cv2
import utils
import numpy as np

img = cv2.imread('opencv.png')

cv2.rectangle(img,(0,100), (500,320), (255,200,80), 10)
cv2.circle(img, (100,500), 10, (255,0,0), 10)
pts = np.array([[100,50], [200,300], [70,20], [500,100]] , np.int32)
cv2.polylines(img, [pts], False, (0,255,0), 3)
utils.imgShow(img)