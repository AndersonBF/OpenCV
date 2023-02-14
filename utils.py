import cv2

def imgShow(img) :
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()