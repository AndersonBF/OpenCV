import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Construir a lookup table para aumentar o valor dos pixels
increase_table = np.clip(UnivariateSpline(x=[0, 64, 128, 255], y=[0, 75, 155, 255])(range(256)), 0, 255).astype(np.uint8)

# Construir a lookup table para diminuir o valor dos pixels
decrease_table = np.clip(UnivariateSpline(x=[0, 64, 128, 255], y=[0, 45, 95, 255])(range(256)), 0, 255).astype(np.uint8)

def applySharpening(image, kernel):
    # Aplicar o filtro de sharpening com o kernel fornecido
    output_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return output_image

# Carregar a imagem
image = cv2.imread("imagem3.png")
if image is None:
    print("Erro: A imagem não foi encontrada.")
    exit()

# Definir os kernels de sharpening
sharpening_kernel0 = np.array([[-1, -1, -1], 
                               [-1, 9, -1], 
                               [-1, -1, -1]])

sharpening_kernel1 = np.array([
    [0, -1,  0],
    [-1, 5, -1],
    [0, -1,  0]
])

# Kernel mais leve
sharpening_kernel_light = np.array([
    [0, -0.5,  0],
    [-0.5, 3, -0.5],
    [0, -0.5,  0]
])

# Aplicar os três filtros
output_image0 = applySharpening(image, sharpening_kernel0)
output_image1 = applySharpening(image, sharpening_kernel1)
output_image_light = applySharpening(image, sharpening_kernel_light)

# Exibir as imagens lado a lado
plt.figure(figsize=[20, 10])
plt.subplot(141)
plt.imshow(image[:, :, ::-1])
plt.title("Imagem Original")
plt.axis('off')

plt.subplot(142)
plt.imshow(output_image0[:, :, ::-1])
plt.title("Sharpening Kernel 0 (9 no centro)")
plt.axis('off')

plt.subplot(143)
plt.imshow(output_image1[:, :, ::-1])
plt.title("Sharpening Kernel 1 (5 no centro)")
plt.axis('off')

plt.subplot(144)
plt.imshow(output_image_light[:, :, ::-1])
plt.title("Sharpening Leve (3 no centro)")
plt.axis('off')

plt.show()