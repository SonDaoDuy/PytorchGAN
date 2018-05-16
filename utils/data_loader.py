import sys
import os
import glob
import shutil
import random as rd
import numpy as np
from skimage import io, transform
from matplotlib import pylab as plt
from tqdm import tqdm
from PIL import Image
import pdb

class Resize(object):
	#  assume image  as H x W x C numpy array
	def __init__(self, output_size):
		assert isinstance(output_size, int)
		self.output_size = output_size

	def __call__(self, image):
		h, w = image.shape[:2]
		if h > w:
			new_h, new_w = self.output_size, int(self.output_size * w / h)
		else:
			new_h, new_w = int(self.output_size * h / w), self.output_size

		resized_image = transform.resize(image, (new_h, new_w))
	
		if h>w:
			diff = self.output_size - new_w
			if diff%2 == 0:
				pad_l = int(diff/2)
				pad_s = int(diff/2)
			else:
				pad_l = int(diff/2)+1
				pad_s = int(diff/2)

			padded_image = np.lib.pad(resized_image, ((0,0), (pad_l,pad_s), (0,0)), 'edge')

		else:
			diff = self.output_size - new_h
			if diff%2==0:
				pad_l = int(diff/2)
				pad_s = int(diff/2)
			else:
				pad_l = int(diff/2)+1
				pad_s = int(diff/2)

			padded_image = np.lib.pad(resized_image, ((pad_l,pad_s), (0,0),  (0,0)), 'edge')

		return padded_image

class CaffeCrop(object):
	#This class take the same behavior as sensenet
	def __init__(self, phase):
		assert(phase=='train' or phase=='test')
		self.phase = phase

	def __call__(self, img):
		# pre determined parameters
		final_size = 224
		final_width = final_height = final_size
		crop_size = 110
		crop_height = crop_width = crop_size
		crop_center_y_offset = 15
		crop_center_x_offset = 0
		if self.phase == 'train':
			scale_aug = 0.02
			trans_aug = 0.01
		else:
			scale_aug = 0.0
			trans_aug = 0.0

		# computed parameters
		randint = rd.randint
		scale_height_diff = (randint(0,1000)/500-1)*scale_aug
		crop_height_aug = crop_height*(1+scale_height_diff)
		scale_width_diff = (randint(0,1000)/500-1)*scale_aug
		crop_width_aug = crop_width*(1+scale_width_diff)


		trans_diff_x = (randint(0,1000)/500-1)*trans_aug
		trans_diff_y = (randint(0,1000)/500-1)*trans_aug


		center = ((img.width/2 + crop_center_x_offset)*(1+trans_diff_x),
				(img.height/2 + crop_center_y_offset)*(1+trans_diff_y))


		if center[0] < crop_width_aug/2:
			crop_width_aug = center[0]*2-0.5
			print(1)
		if center[1] < crop_height_aug/2:
			crop_height_aug = center[1]*2-0.5
			print(2)
		if (center[0]+crop_width_aug/2) >= img.width:
			crop_width_aug = (img.width-center[0])*2-0.5
			print(3)
		if (center[1]+crop_height_aug/2) >= img.height:
			crop_height_aug = (img.height-center[1])*2-0.5
			print(4)

		crop_box = (center[0]-crop_width_aug/2, center[1]-crop_height_aug/2,
					center[0]+crop_width_aug/2, center[1]+crop_width_aug/2)

		mid_img = img.crop(crop_box)
		res_img = mid_img.resize( (final_width, final_height) )
		return res_img

def data_loader_ijba(data_place, img_list_file):
	imgs = []
	caffe_crop = CaffeCrop('test')
	with open(img_list_file, 'r') as imf:
		for line in imf:
			record = line.strip().split()
			images_name = record[0].split('/')[-1]
			img_path, yaw = os.path.join(data_place, images_name), float(record[1])
			imgs.append(img_path)
	
	rsz = Resize(110)
	images_size = len(imgs)
	images = np.zeros((images_size, 110, 110, 3))
	count = 0
	for img_file in imgs:
		#img = io.imread(img_file)
		img = Image.open(img_file).convert("RGB")
		img = caffe_crop(img)
		img = np.array(img)
		img = rsz(img)
		images[count] == img
		count += 1

	#[0,255] -> [-1,1]
	images = images *2 - 1
	# RGB -> BGR
	images = images[:,:,:,[2,1,0]]
	# B x H x W x C-> B x C x H x W
	images = images.transpose(0, 3, 1, 2)

	channel_num = 3

	return images, channel_num


def data_loader(data_place, model_type):
	image_dir = os.path.join(data_place, 'cfp-dataset', 'Data', 'Images')
	split_dir = os.path.join(data_place, 'cfp-dataset', 'Protocol', 'Split', 'FP')

	rsz = Resize(110)

	Indv_dir = []
	for x in os.listdir(image_dir):
		if os.path.isdir(os.path.join(image_dir, x)):
			Indv_dir.append(x)

	test_dir = []
	for x in os.listdir(split_dir):
		if os.path.isdir(os.path.join(split_dir, x)):
			test_dir.append(x)

	Indv_dir=np.sort(Indv_dir)
	test_dir = np.sort(test_dir)

	if model_type == 'Front':
		images = np.zeros((5000, 110, 110, 3))
		id_labels = np.zeros(5000)
	elif model_type == 'Profile':
		images = np.zeros((2000,110,110,3))
		id_labels = np.zeros(5000)

	count = 0

	gray_count = []
	gray_img = []

	#read all images and save in frontal and profile
	for i in tqdm(range(len(Indv_dir))):
		Frontal_dir = os.path.join(image_dir, Indv_dir[i], 'frontal')
		Profile_dir = os.path.join(image_dir, Indv_dir[i], 'profile')

		front_img_files = os.listdir(Frontal_dir)
		prof_img_files = os.listdir(Profile_dir)

		if model_type == 'Front':
			for img_file in front_img_files:
				img = io.imread(os.path.join(Frontal_dir, img_file))
				if len(img.shape)==2:
					print(os.path.join(Frontal_dir, img_file))
					gray_img.append(os.path.join(Frontal_dir, img_file))
					gray_count.append(count)
					id_labels[count] = i
					count += 1
					continue
				img_rsz = rsz(img)
				images[count] = img_rsz
				id_labels[count] = i
				count = count + 1

		if model_type == 'Profile':
			for img_file in prof_img_files:
				img = io.imread(os.path.join(Profile_dir, img_file))
				if len(img.shape)==2:
					gray_img.append(os.path.join(Frontal_dir, img_file))
					gray_count.append(count)
					id_labels[count] = i
					count += 1
					continue
				img_rsz = rsz(img)
				images[count] = img_rsz
				id_labels[count] = i
				count = count + 1

	id_labels = id_labels.astype('int64')

	for i in range(len(gray_img)):
		img = io.imread(gray_img[i])
		index = gray_count[i]
		images[index] = rsz(img)


	#[0,255] -> [-1,1]
	images = images *2 - 1
	# RGB -> BGR
	images = images[:,:,:,[2,1,0]]
	# B x H x W x C-> B x C x H x W
	images = images.transpose(0, 3, 1, 2)

	Nd = 500
	channel_num = 3

	return images, id_labels, Nd, channel_num
