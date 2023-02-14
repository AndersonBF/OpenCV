import cv2

#img = cv2.imread('./opencv.png')
#print(img)
#print(img.shape)
#print(type(img))
#  fazer a imagem aparecer, e se apertar uma tecla, destroi
#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#abre a imagem em tons de cinza, mostra ela 
img = cv2.imread('opencv.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#escreve ela no disco com outro nome e tipo
cv2.imwrite('output.jpg', img)