import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = cv2.imread("Imagens\\img.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("Imagens\\img2.png", cv2.IMREAD_GRAYSCALE)
result_and = cv2.bitwise_and(img, img2, mask=None)
result_or = cv2.bitwise_or(img, img2, mask=None)
result_xor = cv2.bitwise_xor(img, img2, mask=None)

plt.figure(figsize=[10, 5])
plt.subplot(151)
plt.imshow(img, cmap='gray')
plt.title("Imagem Original")
plt.axis('off')
plt.subplot(152)
plt.imshow(img2, cmap='gray')
plt.title("Imagem Original")
plt.axis('off')
plt.subplot(153)
plt.imshow(result_and, cmap='gray')
plt.title("Imagem AND")
plt.axis('off')
plt.subplot(154)
plt.imshow(result_or, cmap='gray')
plt.title("Imagem OR")
plt.axis('off')
plt.subplot(155)
plt.imshow(result_xor, cmap='gray')
plt.title("Imagem XOR")
plt.axis('off')

plt.show()