import cv2
import numpy as np
import pyautogui as u
from matplotlib import pyplot as plot
import time
from time import sleep

W_HEIGHT = 1050
MINIMAP_SIZE = 256
MINIMAP_CONTROL_SIZE = 24


def _show_rgb(img):
    plot.imshow(img)
    plot.show()

# TODO make benchmark before and after resolve todo's
def window_rect(format=0):
    '''
    0 - list
    1 - pair
    3 - left, top, width, height
    '''
    # make search at single screenshot, don't use locateOnScreen twice
    w_start_img, w_end_img = (
        'assets/window_marker.png', 'assets/window_cross.png')
    w_start, w_end = u.locateOnScreen(w_start_img), u.locateOnScreen(w_end_img)
    window_bounding_rect = (
        w_start[0], w_start[1] + w_start[3], w_end[0] + w_end[2], W_HEIGHT + w_end[3])
    if format == 0:
        return window_bounding_rect
    if format == 1:
        return (window_bounding_rect[0:2], window_bounding_rect[2:4])
    if format == 2:
        w = window_bounding_rect
        width = w[2] - w[0]
        height = w[3] - w[1]
        return (w[0], w[1], width, height)


rect = window_rect(format=2)
st = time.time()


def get_minimap(img):
    '''
    img - PIL Image type
    1 screen
    2 to cv2
    3 crop minimap
    4 show minimap
    256 x 24 - controlls
    '''
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    shape = img.shape
    x = shape[1] - MINIMAP_SIZE
    w = shape[1]
    y = 0 + MINIMAP_CONTROL_SIZE
    h = y + MINIMAP_SIZE
    return img[y:h, x:w]


# benchmark
# for i in range(0,100):
id = 0
while True:
    sleep(0.25)
    img = u.screenshot(region=rect)
    roi = get_minimap(img)
    cv2.imwrite('assets/data/mained_' + str(id) + '.png', roi)
    id = id + 1

print('exec_time: ' + str(time.time() - st))
# _show_rgb(roi)
