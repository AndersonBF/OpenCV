import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = cv2.imread("imagem3.png", cv2.IMREAD_COLOR)
#img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


matrix = np.ones(img.shape, dtype="uint8") * 20

img_bright = cv2.add(img, matrix)
img_dark = cv2.subtract(img, matrix)

plt.figure(figsize=[20, 10])
plt.subplot(131)
plt.imshow(img[:, :, ::-1])
plt.title("Imagem Original")
plt.axis('off')
plt.subplot(132)
plt.imshow(img_bright[:, :, ::-1])
plt.title("Imagem Brilhante")
plt.axis('off')
plt.subplot(133)
plt.imshow(img_dark[:, :, ::-1])
plt.title("Imagem Escura")
plt.axis('off')

plt.show()



