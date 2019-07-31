# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import argparse
import collections as col
import numpy as np
import cv2
import math
from skimage import measure as sci

chunk_size = 64

def parse_args():
	parser = argparse.ArgumentParser(
		description='Simple testing funtion for Monodepthv2 models.')

	parser.add_argument('--orig_image', type=str,
						help='Path to the image used as a baseline.', default="panorama3.jpg")
	parser.add_argument('--alt_image', type=str,
						help='Path to the image being compared to the baseline.', default="")
	parser.add_argument('--threshold', type=float, 
						help='Canny threshold value.', default=10.0)
	parser.add_argument('--range', type=float,
						help='Difference between upper and lower canny thresholds.', default=2.5)
	return parser.parse_args()

def sobel_edge(img):
	dy = np.uint8(abs(cv2.Sobel(img, -1, 0, 1, 5)))
	dx = np.uint8(abs(cv2.Sobel(img, -1, 1, 0, 5)))
	retval, img_edge = cv2.threshold(dy + dx, 8, 255, cv2.THRESH_TOZERO)
	return img_edge

def ebiqa(og_img, alt_img):
	#Rough implementation of EBIQA
	height, width = og_img.shape
	og_img_edges = sobel_edge(og_img)
	alt_img_edges = sobel_edge(alt_img)
	
	og_vectors = []
	alt_vectors = []
	distances = []
	y = 0
	x = 0
	
	#generate vectors for each chunk.
	while y < height - (height % chunk_size):
		while x < width - (width % chunk_size):
			og_vectors.append(calc_vector(og_img_edges, y, x))
			alt_vectors.append(calc_vector(alt_img_edges, y, x))
			x += chunk_size
		x = 0
		y += chunk_size

	#Calculate per vector distances.
	for i in range(0, len(og_vectors)):   
		distances.append(dist(og_vectors[i], alt_vectors[i]))
	
	#Return 1 if images are identical.
	if (max(distances) == 0):
		return 1.0
		
	#Otherwise avg out distances.
	return 1 - ((1 / (len(distances) * max(distances))) * np.sum(distances))

	
def calc_vector(img, y_orig, x_orig):
	vector = [0, 0, 0, 0, 0] #EAG, ALE, PLE, NEP, VHO
	chunk = np.zeros((chunk_size, chunk_size), dtype=bool)
	edge_data = []
	
	#Process chunk.
	for y in range(0,chunk_size):
		for x in range(0,chunk_size):
			if chunk[y, x] == False and img[y_orig + y, x_orig + x] != 0:
				new_edge, chunk = fill_edge(img, chunk, y_orig, x_orig, y, x)
				edge_data.append(new_edge)
				chunk[y, x] = vector[0]    
	
	edge_data = np.array(edge_data)
	if (edge_data.size > 0):
		#Calculate vector info.
		vector[0], nope  = edge_data.shape #Get number of edges
		vector[1] = np.sum(edge_data[:, 1]) / vector[0] #Get avg edge length
		vector[2] = max(edge_data[:, 2]) #Get ~PLE estimate, technically not the definition in the paper.
		vector[3] = np.sum(edge_data[:, 2]) #Get number of edge pixels.
		vector[4] = np.sum(edge_data[:, 0]) #Get number of horizontally aligned edges.
	
	return vector

def fill_edge(img, chunk, y_orig, x_orig, y, x): #Get info on a specfic edge.
	pixelQueue = col.deque()
	pixelQueue.append([y, x])
	edge = [0, 0, 1] #horizontally aligned, length, numpixels, 
	queue_len = 1
	min_x = x
	max_x = x
	min_y = y
	max_y = y

	while (queue_len > 0):
		y, x = pixelQueue.popleft()
		queue_len -= 1
		for y_next in [-1, 0, 1]:
			if 0 <= y + y_next < chunk_size:
				for x_next in [-1, 0, 1]:    
					if 0 <= x + x_next < chunk_size and chunk[y + y_next, x + x_next] == False and img[y_orig + y + y_next, x_orig + x + x_next] != 0:
						chunk[y + y_next, x + x_next] = True
						pixelQueue.append([y + y_next, x + x_next])
						queue_len += 1
						min_x = min(min_x, x)
						min_y = min(min_y, y)
						max_x = max(max_x, x)
						max_y = max(max_y, y)
						edge[2] += 1

	if abs(max_x - min_x) > abs(max_y - min_y):
		edge[0] = 1
	
	edge[1] = dist([min_y, min_x], [max_y, max_x])
					
	return (edge, chunk)

def dist(a, b):
	x = 0

	for i in range(0, len(a)):
		x += math.pow(a[i] - b[i], 2)

	return math.sqrt(x)

if __name__ == '__main__':
	args = parse_args()
	og_img = cv2.imread(args.orig_image, 0)
	alt_img = cv2.imread(args.alt_image, 0)
	
	print("SSIM: " + str(sci.compare_ssim(og_img, alt_img)))
	print("MSE: " + str(sci.compare_mse(og_img, alt_img)))   
	print("EBIQA: " + str(ebiqa(og_img, alt_img)))