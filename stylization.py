import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

def applyStylization(image, display=True):
    output_image = cv2.stylization(image, sigma_s=5, sigma_r=0.55)

    if display:
        plt.figure(figsize=[10, 5])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Imagem Original")
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Imagem Estilizada")
        plt.axis('off')
        plt.show()
        return output_image

image= cv2.imread("imagem3.png")
applyStylization(image)