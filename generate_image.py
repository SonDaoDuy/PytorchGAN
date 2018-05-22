#!/usr/bin/env python
# encoding: utf-8

import os
import easydict
import datetime
from matplotlib import pylab as plt
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from model.version_2 import model_P as Model_P
from model import transfer_GAN as transfer_block
from utils.data_loader import data_loader
import pdb

def Generate_Image_v2(images, model_P, args):
	if args.cuda:
		model_P.cuda()
	Nz = 50
	model_P.train()
	batch_size = images.shape[0]
	fixed_noise = torch.FloatTensor(np.random.uniform(-1,1, (batch_size, Nz)))
	batch_image = torch.FloatTensor(images)
	if args.cuda:
		batch_image = batch_image.cuda()
		fixed_noise = fixed_noise.cuda()
	batch_image = Variable(batch_image)
	fixed_noise = Variable(fixed_noise)
	_, x_f, _ = model_P(batch_image, fixed_noise)

	return convert_image(x_f.data.cpu().numpy())

def Generate_Image(images, P_G_model, T_model, F_G_model, args):
	"""

	"""
	if args.cuda:
		P_G_model.cuda()
		T_model.cuda()
		F_G_model.cuda()

	P_G_model.eval()
	T_model.eval()
	F_G_model.eval()
	#form generated model
	P_G_enc = P_G_model.G_enc_convLayers
	F_G_dec = F_G_model.G_dec_convLayers
	Linear_layer = F_G_model.G_dec_fc

	features = []

	batch_size = images.shape[0]
	batch_image = torch.FloatTensor(images)

	batch_image = batch_image.cuda()

	batch_image = Variable(batch_image)

	# Generatorでイメージ生成
	x = P_G_enc(batch_image)
	x = x.view(-1,320)
	x = T_model(x)
	x = x.view(-1,320)
	x = Linear_layer(x)
	x = x.view(-1,320,6,6)
	generated = F_G_dec(x)

	return convert_image(generated.data.cpu().numpy())

def convert_image(data):

	img = data.transpose(0, 2, 3, 1)+1
	img = img / 2.0
	img = img * 255.
	img = img[:,:,:,[2,1,0]]

	return img.astype(np.uint8)

def show_image(jpg_image, generated_image, id_labels, image_list, n):
	plt.rcParams['figure.figsize'] = (15.0, 15.0)
	for i in range(n):
		plt.subplot(2, n, i+1)
		plt.title('No.:{}, id:{}'.format(image_list[i], id_labels[image_list[i]]))
		plt.imshow(jpg_image[image_list[i]])
		plt.subplot(2, n, n+i+1)
		plt.imshow(generated_image[i])

	axes = plt.gcf().get_axes()
	for ax in axes:
		ax.tick_params(labelbottom="off",bottom="off") # x軸の削除
		ax.tick_params(labelleft="off",left="off") # y軸の削除
		ax.set_xticklabels([]) 

	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
	plt.show()

def main():

	args = easydict.EasyDict({
		"cuda": True,
		"data_place": "./dataset",
		"model_type": "Profile",
		})

	#get input image
	n = 20 # number of image show
	images, id_labels, Nd, channel_num = data_loader(args.data_place, args.model_type)
	jpg_image = convert_image(images)
	image_list = np.random.randint(0,len(images), (1,n))[0]

	#create and load model param
	# P_G_model = single_model.Generator(channel_num)
	# T_model = transfer_block.Generator(320,320)
	# F_G_model = single_model.Generator(channel_num)

	# path_to_front_model = './snapshot/Front/2018-05-10_16-51-09/epoch1000_G.pt'
	# path_to_profile_model = './snapshot/Profile/2018-05-11_09-09-05/epoch1000_G.pt'
	# path_to_transfer_model = './snapshot/Transfer/2018-05-14_10-25-52/epoch100_G.pt'

	# P_G_model = torch.load(path_to_profile_model)
	# T_model = torch.load(path_to_transfer_model)
	# print(T_model)
	# F_G_model = torch.load(path_to_front_model)

	model_P = Model_P.Generator(50,3)
	path_to_model_P = 'C:\\Users\\duyson\\Desktop\\Projects\\FaceNormalize\\PytorchGAN\\snapshot\\Model_P\\2018-05-21_15-43-14\\epoch995_G.pt'
	model_P = torch.load(path_to_model_P)

	#gen img
	#generated_imgs = Generate_Image(images[image_list], P_G_model, T_model, F_G_model, args)
	generated_imgs = Generate_Image_v2(images[image_list], model_P, args)
	#show images
	show_image(jpg_image, generated_imgs, id_labels, image_list, n)


if __name__ == '__main__':
	main()
