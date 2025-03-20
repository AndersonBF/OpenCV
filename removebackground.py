import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# Inicializar a segmentação
selfie_seg = mp.solutions.selfie_segmentation
segment = selfie_seg.SelfieSegmentation(model_selection=1)  # Modelo mais preciso

# Carregar a imagem
image = cv2.imread("imagem.png")
if image is None:
    print("Erro: A imagem não foi encontrada.")
    exit()

# Converter para RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Redimensionar a imagem para o modelo
image_resized = cv2.resize(image_rgb, (256, 256))

# Fazer a segmentação
result = segment.process(image_resized)

# Criar a máscara binária com limiar ajustado
binary_mask = result.segmentation_mask > 0.7

# Redimensionar a máscara de volta para o tamanho original da imagem
binary_mask_resized = cv2.resize(binary_mask.astype(np.uint8), (image.shape[1], image.shape[0]))

# Corrigir a máscara para 3 canais
binary_mask_3 = np.repeat(binary_mask_resized[:, :, np.newaxis], 3, axis=2)

# Aplicar a máscara à imagem original
output_image = np.where(binary_mask_3, image_rgb, 255)  # Substituir o fundo por branco (255)

# Exibir a imagem original e a imagem com fundo removido em tamanho menor
plt.figure(figsize=[10, 10])  # Reduzir o tamanho da figura
plt.subplot(121)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(122)
plt.imshow(output_image)
plt.title("Output Image")
plt.axis('off')

plt.show()

