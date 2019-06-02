from image  import extrude_arrow, build_ranges, cvt_points2vectors
import cv2
import numpy as np
import csv

FILE_PREFFIX = 'mained_'
PATH = 'assets/data/'
MAX_FILES = 255

mained_ranges = []

def mock_mained_ranges():
	for i in range(0, MAX_FILES):
		img = cv2.imread(PATH + FILE_PREFFIX + str(i) + '.png')
		img = extrude_arrow(img)
		points = np.transpose(np.nonzero(img))  # get all white points
		x_points, y_points = cvt_points2vectors(points) # format to single vectors
		ranges = build_ranges(x_points)
		mained_ranges.append(ranges)


def write_mained_ranges():
	f = 'assets/data/text/mained_ranges.csv'
	f = open(f, 'w')
	csvFile = csv.writer(f)
	for i, r in enumerate(mained_ranges):
		csvFile.writerow([i])
		if r: 
			for el in r:
				csvFile.writerow([''] + list(el))
		# csvFile.writerow(r)
	f.close()

def write_arrow_ranges_rest():
	mock_mained_ranges()
	write_mained_ranges()

write_arrow_ranges_rest()
