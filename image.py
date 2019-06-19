import cv2
import numpy as np
from matplotlib import pyplot as plot
import draw_utils as utils
import math

import time


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
    # move to const
    lower = np.array([20, 190, 0])  # lower arrow yellow color
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

def full_arrow_entry(ranges, inserts):
    if ranges is None: return
    subrange = []
    is_assigned_range = False
    subrange_points = 0
    for r in ranges:
        if not subrange:
            subrange = [r[1], r[2]]
        else:
            if subrange[1] != r[1] and is_assigned_range:
                is_assigned_range = subrange_points > inserts - subrange_points
            if subrange[1] == r[1]:
                subrange[1] = r[2]
                subrange_points += r[0]
                is_assigned_range = True
            elif not is_assigned_range:
                subrange = [r[1], r[2]]

    return subrange

test_img_path = 'assets/data/mained_0.png';
res = cv2.imread(test_img_path)

def _is_point_in_area(point, area):
    return point >= area[0] and point <= area[1]

def arrow_in_area(x_points, y_points, area):
    return [(x,y) for x, y in zip(x_points, y_points) if _is_point_in_area(x, area)]

def arrow_triangle(arrow):
    arrow_x = [x for x, _ in arrow]
    arrow_y = [y for _, y in arrow]
    triangle = {
        arrow[arrow_x.index(min(arrow_x))],
        arrow[arrow_x.index(max(arrow_x))],
        arrow[arrow_y.index(min(arrow_y))],
        arrow[arrow_y.index(max(arrow_y))]
    }
    triangle = list(triangle)
    if len(triangle) == 3:
        return triangle
    else:
        return None

# cv2 image
def get_arrow_points(image):
    image = extrude_arrow(image)
    points = np.transpose(np.nonzero(image))  # get all white points
    x_points, y_points = cvt_points2vectors(points) # format to single vectors
    
    ranges = build_ranges(x_points)
    arrow_area = full_arrow_entry(ranges, len(x_points))
    if arrow_area is None:
        return None
    arrow = arrow_in_area(x_points, y_points, arrow_area)
    return arrow_triangle(arrow)

arrow = get_arrow_points(res)

def _vector_length(a):
    return np.sqrt(a[0]**2 + a[1]**2)

def _line_length(a, b):
    return np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)

def _vector(a,b):
    return b[0] - a[0], b[1] - a[1]

def _sclr_multiply(a, b):
    return a[0]*b[0] + a[1]*b[1]

def get_angle(a, b):
    va = _vector(a[0], a[1])
    vb = _vector(b[0], b[1])
    vab = _sclr_multiply(va, vb)

    mod_a = _vector_length(va)
    mod_b = _vector_length(vb)
    cos = vab / (mod_a * mod_b)
    return cos


def _k(p1, p2):
    return (p1[1] - p2[1]) / (p1[0] - p2[0])

def tan_f(k1, k2):
    return (k2 - k1) / (1 + k2*k1)


def _degrees(rad):
    return math.degrees(rad)

def min_triangle_angle(arrow):
    a = arrow[0]
    b = arrow[1]
    c = arrow[2]
    abc = [
        _degrees(abs(tan_f(_k(a, b), _k(b, c)))),
        _degrees(abs(tan_f(_k(b, c), _k(c, a)))),
        _degrees(abs(tan_f(_k(c, a), _k(a, b))))
    ]
    
    min_angle = abc.index(min(abc))
    angle_point = min_angle + 1 if min_angle < 2 else 0
    return angle_point

def opposite_point(arrow, acute):
    arrow.remove(arrow[acute])
    oppsite_mid_x = int(round((arrow[0][0] + arrow[1][0]) / 2))
    oppsite_mid_y = int(round((arrow[0][1] + arrow[1][1]) / 2))
    return oppsite_mid_x, oppsite_mid_y

def pointer_angle(arrow):
    acute = min_triangle_angle(arrow)
    b = opposite_point(arrow.copy(), acute)
    c = b[0], arrow[acute][1]
    a = arrow[acute]
    ab = _line_length(b, a)
    bc = _line_length(b, c)
    ac = _line_length(a, c)

    tan_a = bc / ac
    return _degrees(tan_a), (a, b, c), (ab, ac, bc)

# res = cv2.line(res, a, b, [255,0,0], 1)
# res = cv2.line(res, a, c, [255,200,0], 1)
# res = cv2.line(res, b, c, [255,200,0], 1)

# _show_rgb(res)
