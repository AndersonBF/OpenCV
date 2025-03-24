import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = cv2.imread("imagem3.png", cv2.IMREAD_COLOR)

matrix1 = np.ones(img.shape, dtype="uint8") * .7
matrix2 = np.ones(img.shape, dtype="uint8") * 1.3

img_lower = np.uint8(cv2.multiply(np.float64(img), matrix1))
img_higher = np.uint8(np.clip(cv2.multiply(np.float64(img), matrix2), 0, 255))

plt.figure(figsize=[20, 10])
plt.subplot(131)
plt.imshow(img[:, :, ::-1])
plt.title("Imagem Original")
plt.axis('off')
plt.subplot(132)
plt.imshow(img_lower[:, :, ::-1])
plt.title("Imagem Mais Escura")
plt.axis('off')
plt.subplot(133)
plt.imshow(img_higher[:, :, ::-1])
plt.title("Imagem Mais Clara")
plt.axis('off')
plt.show()


