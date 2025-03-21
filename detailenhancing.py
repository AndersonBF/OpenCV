import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

def applyDetailEnhancing(image, display=True):
    output_image = cv2.detailEnhance(image, sigma_s=5, sigma_r=0.05)

    if display:
        plt.figure(figsize=[5, 5])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Imagem Original")
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Imagem com detalhes realçados")
        plt.axis('off')

        plt.show()
    else:
        return output_image
    
image = cv2.imread("imagem2.png")
applyDetailEnhancing(image)
