import cv2
import utils

img = cv2.imread('opencv.png')


#imagens, cordenadas de inicio, cord fim,cor
cv2.line(img, (0,0), (150,150), (205,55,255), 15)
utils.imgShow(img)