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
from utils.data_loader import data_loader, make_pair_data
import pdb
from utils.DataAugmentation import FaceIdPoseDataset_v2, Resize, RandomCrop
from torchvision import transforms
from torch.utils.data import DataLoader

def Generate_Image_v2(images, model_P, Nz, args):
	# batch_image = images[0]
	batch_size = images.shape[0]
	fixed_noise = torch.FloatTensor(np.random.uniform(-1,1, (batch_size, Nz)))
	batch_image = torch.FloatTensor(images.float())
	if args.cuda:
		batch_image = batch_image.cuda()
		fixed_noise = fixed_noise.cuda()
	batch_image = Variable(batch_image)
	fixed_noise = Variable(fixed_noise)
	_, x_f, _ = model_P(batch_image, fixed_noise)
	print(x_f)

	return convert_image(x_f.data.cpu().numpy())

def convert_image(data):

	img = data.transpose(0, 2, 3, 1)+1
	img = img / 2.0
	img = img * 255.
	img = img[:,:,:,[2,1,0]]

	return img.astype(np.uint8)

def show_image(input_img, front_img, generated_image, n):

	r, c = 4, 8
	G_imgs = convert_image(input_img.numpy())
	fig, axs = plt.subplots(r, c)
	cnt = 0
	for i in range(r):
		for j in range(c):
			axs[i,j].imshow(G_imgs[cnt])
			axs[i,j].axis('off')
			cnt += 1
	fig.savefig("./results/input_%d.png" % (n))

	front_imgs = convert_image(front_img.numpy())
	fig, axs = plt.subplots(r, c)
	cnt = 0
	for i in range(r):
		for j in range(c):
			axs[i,j].imshow(front_imgs[cnt])
			axs[i,j].axis('off')
			cnt += 1
	fig.savefig("./results/front_%d.png" % (n))

	gen_imgs = generated_image
	fig, axs = plt.subplots(r, c)
	cnt = 0
	for i in range(r):
		for j in range(c):
			axs[i,j].imshow(gen_imgs[cnt])
			axs[i,j].axis('off')
			cnt += 1
	fig.savefig("./results/gen_%d.png" % (n))
	plt.close()

def main():

	args = easydict.EasyDict({
		"cuda": True,
		"data_place": "./dataset",
		"model_type": "Profile",
		})

	#get input image
	n = 20 # number of image show
	#images, id_labels, Nd, channel_num = data_loader(args.data_place, args.model_type)
	#jpg_image = convert_image(images)
	#image_list = np.random.randint(0,len(images), (1,n))[0]

	#load image
	protocol_dir = './dataset/cfp-dataset/Protocol/Split'
	pair_type = 'FP'
	images_f, id_labels_f, Nd, channel_num = data_loader(args.data_place, 'Front')
	images_p, id_labels_p, Nd, channel_num = data_loader(args.data_place, 'Profile')

	#make pair data
	final_images_p, final_images_f= make_pair_data(images_p, images_f, protocol_dir, pair_type)

	model_P = Model_P.Generator(50,3)
	path_to_model_P = './snapshot/Model_P/l1_loss_remove_emb/epoch24_G.pt'
	model_P = torch.load(path_to_model_P)

	#gen img
	transformed_data = FaceIdPoseDataset_v2(final_images_p, final_images_f, final_images_p, final_images_p,
		transforms=transforms.Compose([
				Resize((110, 110)),
				RandomCrop((96,96))
				]))
	dataloader = DataLoader(transformed_data, batch_size=32, shuffle=False, pin_memory=True)
	if args.cuda:
		model_P.cuda()
	Nz = 50
	model_P.eval()
	gen_img = []
	for i, batch_img in enumerate(dataloader):
		input_img = batch_img[0]
		#print(input_img)
		front_img = batch_img[1]
		generated_imgs = Generate_Image_v2(input_img, model_P, Nz, args)
		show_image(input_img, front_img, generated_imgs, i)


if __name__ == '__main__':
	main()
