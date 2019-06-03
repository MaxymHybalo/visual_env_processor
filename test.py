from image  import extrude_arrow, build_ranges, cvt_points2vectors
import cv2
import numpy as np
import csv

FILE_PREFFIX = 'mained_'
PATH = 'assets/data/'
MAX_FILES = 255

mained_ranges = []
clear_ranges = []
points_count = []

def mock_mained_ranges():
	for i in range(0, MAX_FILES):
		img = cv2.imread(PATH + FILE_PREFFIX + str(i) + '.png')
		img = extrude_arrow(img)
		points = np.transpose(np.nonzero(img))  # get all white points
		x_points, y_points = cvt_points2vectors(points) # format to single vectors
		ranges = build_ranges(x_points)
		mained_ranges.append(ranges)
		if ranges is not None:
			clear_ranges.append((i, ranges))
		points_count.append(len(points))

def _ranges(csf):
	for i, r in clear_ranges:
		csf.writerow([i])
		if r: 
			for el in r:
				csf.writerow([''] + list(el))

def _ranges_count(csf):
	for i, r in enumerate(mained_ranges):
		inserts = 0
		if r: 
			for el in r:
				inserts += el[0]
		csf.writerow([i,'',inserts, '', points_count[i]])

def _ranges_clear(csf):
	for i, r in enumerate(mained_ranges):
		csf.writerow([i])
		if r: 
			for el in r:
				csf.writerow([''] + list(el))

def _ranges_count_clear(csf):
	for i, r in clear_ranges:
		inserts = 0
		if r: 
			for el in r:
				inserts += el[0]
		csf.writerow([i,'',inserts, '', points_count[i]])

def write_data(path, selecor):
	f = path
	f = open(f, 'w')
	csvFile = csv.writer(f)
	selecor(csvFile)
	f.close()


def write_arrow_ranges_rest():
	mock_mained_ranges()
	write_data(PATH + 'text/mained_ranges.csv', _ranges)
	write_data(PATH + 'text/mained_ranges_count.csv', _ranges_count)
	write_data(PATH + 'text/mained_ranges_clear.csv', _ranges_clear)
	write_data(PATH + 'text/mained_ranges_count_clear.csv', _ranges_count_clear)

write_arrow_ranges_rest()
