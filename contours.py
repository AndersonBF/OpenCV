import numpy as np
import cv2 as cv

im = cv.imread("imagem2.png")
assert im is not None, "file could not be read, check with os.path.exists()"

imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(imgray, 127, 255, 0)

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

# Desenhar os contornos na imagem original
cv.drawContours(im, contours, -1, (0, 255, 0), 3)

cv.imshow("Contornos", im)

cv.waitKey(0)
cv.destroyAllWindows()