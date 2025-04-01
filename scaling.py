import numpy as np
import cv2 as cv

# Carregar a imagem
img = cv.imread("imagem2.png")
assert img is not None, "file could not be read, check with os.path.exists()"

# Redimensionar a imagem
res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

# Salvar a imagem redimensionada na mesma pasta
cv.imwrite("imagem2_redimensionada.png", res)

print("Imagem redimensionada salva como 'imagem2_redimensionada.png'")