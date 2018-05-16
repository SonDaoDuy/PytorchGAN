#!/usr/bin/env python
# encoding: utf-8

# Data Augmentation class which is used with DataLoader
# Assume numpy array face images with B x C x H x W  [-1~1]

import scipy as sp
import numpy as np
from skimage import transform
from torchvision import transforms
from torch.utils.data import Dataset
import pdb

class IJBADataset(object):
	"""docstring for IJBADataset"""
	def __init__(self, images, transforms=None):
		self.images = images
		self.transforms = transforms

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		image = self.images[idx]
		if self.transforms:
			image = self.transforms(image)
		return image
		

class FaceIdPoseDataset(Dataset):
	"""

	"""
	def __init__(self, images, IDs, transforms=None):
		
		self.images = images
		self.IDs = IDs
		self.transforms = transforms

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		image = self.images[idx]
		ID = self.IDs[idx]
		if self.transforms:
			image = self.transforms(image)

		return [image, ID]

class PFFeatDataset(Dataset):
	"""

	"""
	def __init__(self, front_feat, profile_feat, IDs, transforms=None):
		
		self.front_feat = front_feat
		self.profile_feat = profile_feat
		self.IDs = IDs
		self.transforms = transforms

	def __len__(self):
		return len(self.front_feat)

	def __getitem__(self, idx):
		front_feat = self.front_feat[idx]
		profile_feat = self.profile_feat[idx]
		ID = self.IDs[idx]
		if self.transforms:
			front_feat = self.transforms(front_feat)
			profile_feat = self.transforms(profile_feat)

		return [front_feat, profile_feat, ID]

class Resize(object):
	"""docstring for Resize"""
	def __init__(self, output_size):
		assert isinstance(output_size, (tuple))
		self.output_size = output_size
		
	def __call__(self, image):
		new_h, new_w = self.output_size
		pad_width = int((new_h - image.shape[1]) / 2)
		resized_image = np.lib.pad(image, ((0,0), (pad_width,pad_width),(pad_width,pad_width)), 'edge')

		return resized_image

class RandomCrop(object):

	#  assume image  as C x H x W  numpy array

	def __init__(self, output_size):
		assert isinstance(output_size, tuple)
		assert len(output_size) == 2
		self.output_size = output_size

	def __call__(self, image):
		h, w = image.shape[1:]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		cropped_image = image[:, top:top+new_h, left:left+new_w]

		return cropped_image