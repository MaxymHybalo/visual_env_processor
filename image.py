import cv2
import numpy as np
from matplotlib import pyplot as plot
import draw_utils as utils

IMG = 'test_image_3.png'
ARROW_BOUNDARY = 10

def _show_rgb(img):
    plot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plot.show()


def _show_gray(img):
    plot.imshow(img, cmap="gray")
    plot.show()


def _show_range(imgs):
    for id, img in enumerate(imgs):
        plot.subplot(1, len(imgs), id +
                     1), plot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
    lower = np.array([21, 190, 0])  # lower arrow yellow color
    upper = np.array([28, 255, 255])  # upper
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    img = cv2.bitwise_and(img, img, mask=mask)  # filter
    img = _to_gray(img)  # to gray
    # threshold noise, probably
    _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    return img


def draw_corners(img):  # gray channel img
    gray = np.float32(img)
    corners = cv2.goodFeaturesToTrack(gray, 200, 0.01, 4)
    corners = np.int0(corners)
    img = _gray_to_rgb(img)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 3, 255, 1)
    return img

def cvt_points2vectors(points):
    def yf(point): return point[0]
    def xf(point): return point[1]

    y_points = np.array([yf(yi) for yi in points])
    x_points = np.array([xf(xi) for xi in points])
    return x_points, y_points

def _count_inserts(points, start, end, precise_end):
    inserts = 0
    for p in points:
        end_state = p <= end if precise_end else p < end
        if (p >= start and end_state):
            inserts += 1
    # print('inserts ', inserts ,' start ', start, ' end ', end)
    return  inserts

def _build_ranges(points, start, delta):
    ranges = []
    start_0 = start
    for i in range(1, ARROW_BOUNDARY + 1):
        end = start_0 + delta * i
        inserts = _count_inserts(points, start, end, i == ARROW_BOUNDARY)
        if (inserts > 0):
            ranges.append((inserts, start, end))
        start = end
    # print('ranges ', ranges)
    return ranges

def build_ranges(axis):
    if (len(axis) <= 25):
        return None
    a_min = min(axis)
    arange = max(axis) - a_min
    delta = arange / ARROW_BOUNDARY
    return _build_ranges(axis, a_min, delta)

examples = []
# for i in range(1, 2):
# test_img_path = 'assets/arrow_test/map_5.png'
test_img_path = 'assets/data/mained_104.png';
res = cv2.imread(test_img_path)
res = extrude_arrow(res)

# res = draw_corners(res)
# _show_gray(res)

points = np.transpose(np.nonzero(res))  # get all white points
x_points, y_points = cvt_points2vectors(points) # format to single vectors

# for i in range(0, len(x_points)):
    # print(x_points[i], y_points[i])

# ranges = build_ranges(x_points)

# examples.append(res)
# _, ax = plot.subplots()
# for p in points:
#     ax.scatter(p[1], p[0])
# plot.show()

# for i in range(1,7):
# test_img_path = 'arrow_test/map_' + str(i) + '.png'
# res = cv2.imread(test_img_path)
# examples.append(res)

# _show_range(examples)

# cv2.cvtColor(np.uint8([[[0,0,0]]]), cv2.COLOR_BGR2HSV)
