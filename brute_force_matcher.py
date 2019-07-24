import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

FILE_PREFFIX = 'mained_'
PATH = 'assets/data/'
MAX_FILES = 122

test_file = PATH + FILE_PREFFIX + '0.png'

arrow = cv2.imread('assets/arrow.png', 0)
test = cv2.imread(test_file, 0)
# research ORB_create
# orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=10)
orb = cv2.ORB_create(nfeatures=100, scoreType=cv2.ORB_HARRIS_SCORE, edgeThreshold=8)
kp1, des1 = orb.detectAndCompute(arrow, None)
kp2, des2 = orb.detectAndCompute(test, None)

print(des1)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x:x.distance)

# print(matches)
result = cv2.drawMatches(arrow, kp1, test, kp2, matches, None, flags=2)

plt.imshow(result)
plt.show()