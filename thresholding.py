import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = cv2.imread("imagem.png", cv2.IMREAD_GRAYSCALE)
retval, img_thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

retval, img_thresh2 = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)

img_thresh_adap = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)

plt.figure(figsize=(10, 10));plt.subplot(221);plt.title("Original");plt.imshow(img, cmap="gray")
plt.subplot(222);plt.title("Thresholding");plt.imshow(img_thresh, cmap="gray")
plt.subplot(223);plt.title("Thresholding 2");plt.imshow(img_thresh2, cmap="gray")
plt.subplot(224);plt.title("Thresholding Adaptativo");plt.imshow(img_thresh_adap, cmap="gray")

plt.show()