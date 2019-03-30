import cv2
import numpy as np
from matplotlib import pyplot as plot
import draw_utils as utils

IMG = 'test_image_3.png'

def _show_rgb(img):
    plot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plot.show()

def _show_gray(img):
    plot.imshow(img, cmap="gray")
    plot.show()

def _show_range(imgs):
    for id, img in enumerate(imgs):
        plot.subplot(1, len(imgs), id+1), plot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plot.show()

def _gray_to_rgb(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def _to_gray(rgb):
    return cv2.cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

def extrude_arrow(img):
    '''
        img should be bgr format
        returns binary image with arrow shape hightlighted
        look at options down for future enhancement
        # not bad circles detection with
        # 1. blur (5,5)
        # 2. param1 - 60, param2 - 35
        # also with errosion 5,5 param1 10
    '''
    lower = np.array([21,190,0]) # lower arrow yellow color
    upper = np.array([28, 255,255]) # upper 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    img = cv2.bitwise_and(img, img, mask=mask) # filter
    img = _to_gray(img) # to gray
    _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY) # threshold noise, probably
    return img

def draw_corners(img): # gray channel img
    gray = np.float32(img)
    corners = cv2.goodFeaturesToTrack(gray, 200, 0.01, 4)
    corners = np.int0(corners)
    img = _gray_to_rgb(img)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x,y), 3, 255, 1)
    return img

examples = []

test_img_path = 'arrow_test/map_3.png'
res = cv2.imread(test_img_path)
res = extrude_arrow(res)
points = np.transpose(np.nonzero(res)) # get all white points
yf = lambda point: point[1]
xf = lambda point: point[0]

y_points = np.array([yf(yi) for yi in points])
x_points = np.array([xf(xi) for xi in points])

def frequences(range):
    min, max = np.min(range), np.max(range)
    R = max - min
    N = np.round(1 + 1.322*np.log10(range.size))
    h = np.round(R / N)
    sort = np.sort(range)
    freqs = np.zeros(int(N))
    step = min
    for i, y in enumerate(freqs):
        step_upper = step + h*(i + 1)
        gt = sort[sort >= step]
        lr = gt[gt <= step_upper]
        freqs[i] = lr.size
        step = step_upper
    return freqs

x_freq = frequences(x_points)
y_freq = frequences(y_points)
print(x_freq, y_freq)

# _, ax = plot.subplots()
# for i, r in enumerate(y_range):
    # print(r)
    # ax.scatter(1,1, s=r, color='r', alpha=1/r)
# plot.show()

# for i in range(1,7):
    # test_img_path = 'arrow_test/map_' + str(i) + '.png'
    # res = cv2.imread(test_img_path)
    # res = extrude_arrow(res)
    # points = np.transpose(np.nonzero(res))
    # print(points)
    # print('========')
    # res = _gray_to_rgb(res)
    # res = draw_corners(res)  
    # examples.append(res)

# _show_range(examples)

# cv2.cvtColor(np.uint8([[[0,0,0]]]), cv2.COLOR_BGR2HSV)