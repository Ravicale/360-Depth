from __future__ import absolute_import, division, print_function

import os
import glob
import argparse
import collections as col
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import cv2
import torch
import math

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from torchvision import transforms, datasets

def run_monodepth(model_name, image_path, ext):
	#Function to predict for a single image or folder of images
	assert model_name is not None, \
		"You must specify the --model_name parameter; see README.md for an example"

	device = torch.device("cpu")

	download_model_if_doesnt_exist(model_name)
	model_path = os.path.join("models", model_name)
	print("-> Loading model from ", model_path)
	encoder_path = os.path.join(model_path, "encoder.pth")
	depth_decoder_path = os.path.join(model_path, "depth.pth")

	# LOADING PRETRAINED MODEL
	print("   Loading pretrained encoder")
	encoder = networks.ResnetEncoder(18, False)
	loaded_dict_enc = torch.load(encoder_path, map_location=device)

	# extract the height and width of image that this model was trained with
	feed_height = loaded_dict_enc['height']
	feed_width = loaded_dict_enc['width']
	filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
	encoder.load_state_dict(filtered_dict_enc)
	encoder.to(device)
	encoder.eval()

	print("   Loading pretrained decoder")
	depth_decoder = networks.DepthDecoder(
		num_ch_enc=encoder.num_ch_enc, scales=range(4))

	loaded_dict = torch.load(depth_decoder_path, map_location=device)
	depth_decoder.load_state_dict(loaded_dict)

	depth_decoder.to(device)
	depth_decoder.eval()

	# FINDING INPUT IMAGES
	if os.path.isfile(image_path):
		# Only testing on a single image
		paths = [image_path]
		output_directory = os.path.dirname(image_path)
	elif os.path.isdir(image_path):
		# Searching folder for images
		paths = glob.glob(os.path.join(image_path, '*.{}'.format(ext)))
		output_directory = image_path
	else:
		raise Exception("Can not find image_path: {}".format(image_path))

	print("-> Predicting on {:d} test images".format(len(paths)))

	# PREDICTING ON EACH IMAGE IN TURN
	with torch.no_grad():
		for idx, image_path in enumerate(paths):

			if image_path.endswith("_disp.jpg"):
				# don't try to predict disparity for a disparity image!
				continue

			# Load image and preprocess
			input_image = pil.open(image_path).convert('RGB')
			original_width, original_height = input_image.size
			input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
			input_image = transforms.ToTensor()(input_image).unsqueeze(0)

			# PREDICTION
			input_image = input_image.to(device)
			features = encoder(input_image)
			outputs = depth_decoder(features)

			disp = outputs[("disp", 0)]
			disp_resized = torch.nn.functional.interpolate(
				disp, (original_height, original_width), mode="bilinear", align_corners=False)

			# Saving numpy file
			output_name = os.path.splitext(os.path.basename(image_path))[0]
			name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
			scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
			np.save(name_dest_npy, scaled_disp.cpu().numpy())

			# Saving colormapped depth image
			disp_resized_np = disp_resized.squeeze().cpu().numpy()
			vmax = np.percentile(disp_resized_np, 95)
			normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
			mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
			colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
			im = pil.fromarray(colormapped_im)

			name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
			im.convert('L').save(name_dest_im)

			print("   Processed {:d} of {:d} images - saved prediction to {}".format(
				idx + 1, len(paths), name_dest_im))

	print('-> Depth Generated!')
	
def parse_args():
	parser = argparse.ArgumentParser(
		description='Simple testing funtion for Monodepthv2 models.')

	parser.add_argument('--image_path', type=str,
						help='path to a test image or folder of images', default="panorama3.jpg")
	parser.add_argument('--ext', type=str,
						help='image extension to search for in folder', default="jpg")
	parser.add_argument('--k_value', type=int,
						help='Number of uniquely colored elements in image', default=6)
	parser.add_argument('--depth_scale', type=int,
						help='Contribution of normalmap to final depth map.', default=700)
	parser.add_argument('--y_norm_offset', type=int,
						help='Value to add/subtract from vertical normals.', default=0)
	parser.add_argument('--x_norm_offset', type=int,
					help='Value to add/subtract from horizontal normals.', default=0)
	parser.add_argument('--skip_ahead', type=bool,
						help='Use segmentation from previous run in depth evaluation. Can save a lot of time with large images', default=False)
	return parser.parse_args()

def cluster_image(img, k):
	height, width, channels = img.shape
	
	#Clear out noise in image for better segmentation.
	print('-> Smoothing Image')
	img = cv2.bilateralFilter(img, -1, 64, 64)  
	
	#Reshape image into k-means friendly form.
	print('-> Beginning Segmentation')
	img = img.reshape((-1,3))
	
	#Convert to np.float32
	img = np.float32(img)
	
	print('   Generating Criteria')
	#Define criteria, number of clusters (K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	print('   K=' + str(k))
	print('   Calculating K-Means')
	ret,label,center=cv2.kmeans(img,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	
	print('   Reshaping Segmented Image')
	#Now convert back into uint8 and original shape.
	center = np.uint8(center)
	img = center[label.flatten()]
	img = img.reshape((height, width, channels))
	
	return img

def add_alpha(img):
	#Adds a blank alpha channel to an image.
	b_vals, g_vals, r_vals = cv2.split(img)
	a_vals = np.zeros(b_vals.shape, dtype=b_vals.dtype)
	img = cv2.merge((b_vals, g_vals, r_vals, a_vals))
	
	return img
	
def segment_image(img):
	height, width, channels = img.shape

	#Apply floodfill to each section such that each 'object' has unique colors.
	print('-> Indexing Objects')
	obj_id = 0
	for y in range(0, height):
		for x in range(0, width):
			if img[y, x, 3] == 0: #Skip already calculated pixels.
				blue = obj_id % 255
				green = (obj_id / 255) % 255
				red = (obj_id / 65025) % 255
				color = np.uint8(np.array([blue, green, red, 255]))
				img = floodfill(img, img, y, x, color)
				obj_id += 1
	print("   Indexing Complete, " + str(obj_id) + " Regions Found")
	return img

def regionalize_colors(img, norms, mask, depth_scale):
	height, width, channels = img.shape
	mask = mask.copy() #Without this images get mutated outside of this scope for reasons?!
	img = img.copy()

	print('-> Getting Per Region depth')
	max_id = 0
	obj_colors = []
	append = obj_colors.append #Obj lookup each iteration is slow.

	#Create a list with each segment to get mean depth, position, and normal direction for each one.
	print("   Calculating data for each region.")
	for y in range(0, height):
		for x in range(0, width):
			curr_id = get_id(mask, y, x)
			curr_col = get_pixel_meta(img, norms, y, x)
			if curr_id == max_id: #If new segment discovered, append to segment list.
				append(curr_col)
				max_id += 1
			else: #Otherwise add to be averaged later.
				obj_colors[curr_id] = list(map(lambda a, b : a + b, obj_colors[curr_id], curr_col))
	print("   Per Region depth found.")
	print("   Painting new image")
	obj_colors = list(map(mean, obj_colors)) #Find average values for each segment.
	#Then calculate final depth values.
	img = np.array(list(map(lambda y : map_scanline(mask, obj_colors, y, depth_scale), list(range(height)))))
	print("   Painting complete.")
	print("   Smoothing result.")
	img = cv2.medianBlur(img, 5)
	return img

def map_scanline(mask, pixel_data, y, depth_scale):
	height, width, channels = mask.shape
	return list(map(lambda x: map_pixel(mask, pixel_data, y, x, depth_scale), list(range(width))))

def map_pixel(mask, pixel_data, y, x, depth_scale):
	height, width, channels = mask.shape
	#Model segment as a plane to find depths.
	#Get segment id.
	curr_id = mask[y, x, 0] + (mask[y, x, 1] * 255) + (mask[y, x, 2] * 65025)
	#Get distance of pixel from segment centroid.
	vert_dist = np.float64(pixel_data[curr_id][3]) - y
	horz_dist = np.float64(pixel_data[curr_id][4]) - x
	#Get normal vector components.
	vert_norm = math.pi * (np.float64(pixel_data[curr_id][1] - 128) / 255)
	horz_norm = math.pi * (np.float64(pixel_data[curr_id][2] - 128) / 255)
	#Calculate distance of pixel depth from centroid depth of segment.
	vert_dist = (math.tan(vert_norm) * (vert_dist / height))
	horz_dist = (math.tan(horz_norm) * (horz_dist / width))
	#Calculate final depth.
	color = min(max(pixel_data[curr_id][0] + (-depth_scale * (vert_dist + horz_dist)), 0.0), 255)
	return np.uint8([color, color, color])
	

def generate_normal_map(depth, offset):
	#Runs a x/y sobel filter to get a normal map of monodepth2 output.
	height, width, channels = depth.shape
	depth_gray = np.float32(depth[..., 0]) / 255
	depth_gray = cv2.GaussianBlur(depth_gray, (33, 33), 33, 0)
	
	dy = cv2.Sobel(depth_gray, -1, 0, 1, 3)
	dy = np.uint8(np.clip((dy - np.min(dy))/np.ptp(dy) * 255 + offset[0], 0, 255))
	
	dx = cv2.Sobel(depth_gray, -1, 1, 0, 3)
	dx = np.uint8(np.clip((dx - np.min(dx))/np.ptp(dx) * 255 + offset[1], 0, 255))
	
	return cv2.merge((np.full((height, width), 0, np.uint8), dx, dy))
	

def floodfill(img, mask, y_orig, x_orig, color):
	mask_region = get_pixel(mask, y_orig, x_orig)
	
	pixelQueue = col.deque()
	pixelQueue.append([y_orig, x_orig])
	queue_len = 1
	img = set_pixel(img, color, y_orig, x_orig)
	while (queue_len > 0):
		y, x = pixelQueue.popleft()
		queue_len -= 1
		for y_next in [-1, 0, 1]:
			for x_next in [-1, 0, 1]:        
				if compare_color(mask, mask_region, y + y_next, x + x_next):
					pixelQueue.append([y + y_next, x + x_next])
					queue_len += 1
					mask[y+y_next, x+x_next, :] += 1
					img = set_pixel(img, color, y + y_next, x + x_next)
	return img

def get_id(img, y, x):
	#Gets unique segment id for a given pixel.
	return img[y, x, 0] + (img[y, x, 1] * 255) + (img[y, x, 2] * 65025)

def compare_color(img, color, y, x):
	height, width, channels = img.shape
	if (y < 0 or y > height-1):
		return False
	if (x < 0 or x > width-1):
		return False
	
	blue = img[y, x, 0] == color[0]
	green = img[y, x, 1] == color[1]
	red = img[y, x, 2] == color[2]
	if channels == 4:
		alpha = img[y, x, 3] == color[3]
	else:
		alpha = True
	
	return blue and green and red and alpha

def set_pixel(img, color, y, x):
	height, width, channels = img.shape
	if (y < 0 or y > height-1):
		return img
	if (x < 0 or x > width-1):
		return img
	
	for channel in range(0, channels):
		img[y, x, channel] = color[channel]
	
	return img

def get_pixel(img, y, x):
	height, width, channels = img.shape
	if (y < 0 or y > height-1):
		return img
	if (x < 0 or x > width-1):
		return img
	
	color = []
	for channel in range(0, channels):
		color.append(img[y, x, channel])
		
	return color

def get_pixel_meta(img, norm, y, x):
	#Used to get loads of metadata for a pixel in a segment.
	height, width, channels = img.shape
	if (y < 0 or y > height-1):
		return img
	if (x < 0 or x > width-1):
		return img
	
	return np.uint64([img[y, x, 0], norm[y, x, 2], norm[y, x, 1], y, x, 1])

def mean(pixel_data):
	return list(map(lambda i : i / pixel_data[5], pixel_data))

if __name__ == '__main__':
	#Create folder to contain cropped images.
	output_dir = os.path.dirname(os.path.realpath(__file__))
	if os.path.exists(output_dir + '/out') != 1:
		print('   Out Directory Not Found, Building New One.')
		os.mkdir(output_dir + '\out')
	output_dir = output_dir + '\out'        
	args = parse_args()
	
	def save_image(name, img):
		output_file = os.path.join(output_dir, name)
		cv2.imwrite(output_file, img)
	print('   Saving files to: ' + output_dir)

	if args.skip_ahead == False:
		pan_img = cv2.imread(args.image_path)
		save_image("pan.jpg", pan_img)
		run_monodepth("mono+stereo_1024x320", output_dir, args.ext)
		
		k_img = cluster_image(pan_img, args.k_value)
		save_image("pan_k_cluster.png", k_img)
		k_img = add_alpha(k_img)
		s_img = segment_image(k_img)
		save_image("pan_segment.png", s_img)
	
		depth = cv2.imread(output_dir + "\pan_disp.jpeg")
		normals = generate_normal_map(depth, (args.y_norm_offset, args.x_norm_offset))
		save_image("pan_norm_disp.jpeg", normals)
		
		dep_img = regionalize_colors(depth, normals, s_img, args.depth_scale)
		save_image("pan_depth.jpeg", dep_img)
	else:
		depth = cv2.imread(output_dir + "\pan_disp.jpeg")
		normals = generate_normal_map(depth, (args.y_norm_offset, args.x_norm_offset))
		save_image("pan_norm_disp.jpeg", normals)
		s_img = cv2.imread(output_dir + "\pan_segment.png")
		dep_img = regionalize_colors(depth, normals, s_img, args.depth_scale)
		save_image("pan_depth.jpeg", dep_img)