from image  import extrude_arrow, build_ranges, cvt_points2vectors, full_arrow_entry, get_arrow_points
import cv2
import numpy as np
import csv

FILE_PREFFIX = 'mained_'
PATH = 'assets/data/'
MAX_FILES = 255

mained_ranges = []
clear_ranges = []
points_count = []
full_range = []
triangles = []

def mock_mained_ranges():
	for i in range(0, MAX_FILES):
		img = cv2.imread(PATH + FILE_PREFFIX + str(i) + '.png')
		triangles.append(get_arrow_points(img))
		img = extrude_arrow(img)
		points = np.transpose(np.nonzero(img))  # get all white points
		x_points, y_points = cvt_points2vectors(points) # format to single vectors
		ranges = build_ranges(x_points)
		mained_ranges.append(ranges)
		if ranges is not None:
			clear_ranges.append((i, ranges))
			full_range.append((i, full_arrow_entry(ranges, len(x_points))))
		points_count.append(len(points))

def draw_arrow_area():
	for i in range(0, MAX_FILES):
		img = cv2.imread(PATH + FILE_PREFFIX + str(i) + '.png')
		triangle = get_arrow_points(img)
		if triangle:
			img = cv2.line(img, triangle[2], triangle[0] ,(0,20,255), 1)
			img = cv2.line(img, triangle[1], triangle[2] ,(0,20,255), 1)
			img = cv2.line(img, triangle[0], triangle[1] ,(0,20,255), 1)
			cv2.imwrite(PATH + 'fined_arrows/' + str(i) + '.png', img)

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
		# print(i)
		if r: 
			for el in r:
				csf.writerow([''] + list(el))
				# print(el)

def _ranges_count_clear(csf):
	for i, r in clear_ranges:
		inserts = 0
		if r: 
			for el in r:
				inserts += el[0]
		csf.writerow([i,'',inserts, '', points_count[i]])

def _full_ranges(csf):
	for i, r in full_range:
		csf.writerow([i, '', r])
		if mained_ranges[i]: 
			for el in mained_ranges[i]:
				csf.writerow([''] + list(el))


def _triangles(csf):
	for i, t in enumerate(triangles):
		if t: 
			csf.writerow([i, '', '', len(t)])
			for el in t:
				csf.writerow([''] + list(el))

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
	write_data(PATH + 'text/full_range.csv', _full_ranges)
	write_data(PATH + 'text/triangles.csv', _triangles)

# write_arrow_ranges_rest()

draw_arrow_area()