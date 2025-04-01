import numpy as np
import cv2 as cv

# Carregar a imagem em escala de cinza
img = cv.imread("imagem2.png", cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# Obter as dimensões da imagem
rows, cols = img.shape

# Criar a matriz de rotação
M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 90, 1)

# Aplicar a rotação
dst = cv.warpAffine(img, M, (cols, rows))

# Salvar a imagem rotacionada
cv.imwrite("imagem2_rotacionada.png", dst)

print("Imagem rotacionada salva como 'imagem2_rotacionada.png'")