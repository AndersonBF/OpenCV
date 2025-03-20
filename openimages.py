import numpy as np
import cv2
from matplotlib import pyplot as plt

def showImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')  # Esconder os eixos
    plt.show()

def getColor(img, x, y):
    return img[y, x, 0], img[y, x, 1], img[y, x, 2]

def setColor(img, x, y, b, g, r):
    img[y, x, 0] = b
    img[y, x, 1] = g
    img[y, x, 2] = r

def main():
    obj_img = cv2.imread("imagem.png")
    if obj_img is None:
        print("Erro: A imagem não foi encontrada.")
        return
    
    altura, largura, canais_de_cor = obj_img.shape
    print("Dimensões da imagem: " + str(largura) + "X" + str(altura))
    print("Canais de cor: " + str(canais_de_cor))
    
    # Recortar uma região da imagem
    pera_img = obj_img[109:180, 220:249]
    showImage(pera_img)

    # Definir novas coordenadas para colar a região recortada


    # Colar a região recortada na nova posição
    obj_img[250: 250 + pera_img.shape[0], 220:220 + pera_img.shape[1]] = pera_img
    showImage(obj_img)

main()