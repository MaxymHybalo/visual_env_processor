from image  import extrude_arrow, build_ranges, cvt_points2vectors, full_arrow_entry, get_arrow_points, min_triangle_angle, pointer_angle, opposite_point
import cv2
import numpy as np
import csv

FILE_PREFFIX = 'mained_horizon_'
PATH = 'assets/data/'
MAX_FILES = 122

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
			angle, points, lengths = pointer_angle(triangle)
			a,b,c = points
			img = cv2.line(img, a, b,(255,0,0), 1)
			img = cv2.line(img, a, c,(0,255,0), 1)
			img = cv2.line(img, b, c,(0,0,255), 1)
			img = cv2.circle(img, a, 1, (100, 100,255), 1)
			img = cv2.putText(img, str(angle), (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)
			img = cv2.putText(img, str(lengths[0]), (0,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
			img = cv2.putText(img, str(lengths[1]), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
			img = cv2.putText(img, str(lengths[2]), (0,75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

			cv2.imwrite(PATH + 'acute_angle/' + str(i) + '.png', img)

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

def _angles(csf):
	for i, t in enumerate(triangles):
		if t: 
			csf.writerow([i, pointer_angle(t)])

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
	write_data(PATH + 'text/angles.csv', _angles)

write_arrow_ranges_rest()

draw_arrow_area()