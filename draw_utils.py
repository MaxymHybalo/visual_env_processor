import cv2

def content_roi(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.boundingRect(img)
    return x, y, w, h