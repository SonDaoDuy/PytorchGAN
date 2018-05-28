#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import datetime
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from model import single_GAN as Model_F
from model.version_2 import model_P as Model_P
from model.version_2.train_model_P import train_model_P
from utils.data_loader import data_loader, make_pair_data
import pdb
from utils.read_bin_cfp import read_bin_cfp, load_feat


def parse_args():
	parser = argparse.ArgumentParser(description='PF_GAN')
	# learning & saving parameterss
	parser.add_argument('-lr', type=float, default=0.0002, help='initial learning rate [default: 0.0002]')
	parser.add_argument('-beta1', type=float, default=0.5, help='adam optimizer parameter [default: 0.5]')
	parser.add_argument('-beta2', type=float, default=0.999, help='adam optimizer parameter [default: 0.999]')
	parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 1000]')
	parser.add_argument('-batch_size', type=int, default=64, help='batch size for training [default: 8]')
	parser.add_argument('-save_dir', type=str, default='snapshot', help='where to save the snapshot')
	parser.add_argument('-save_freq', type=int, default=1, help='save learned model for every "-save-freq" epoch')
	parser.add_argument('-cuda', action='store_true', default=True, help='enable the gpu')
	# data souce
	#parser.add_argument('-random', action='store_true', default=False, help='use randomely created data to run program')
	parser.add_argument('-data_place', type=str, default='./dataset', help='prepared data path to run program')
	# model
	parser.add_argument('-model_type', type=str, default='Model_P', help='train model_P')
	parser.add_argument('-load_model', type=bool, default=False, help='Load existed model')
	parser.add_argument('-split_id', type=int, default=10, help='split data folder for test')
	#parser.add_argument('-images-perID', type=int, default=0, help='number of images per person to input to multi image DR_GAN')
	# option
	#parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot(snapshot/{Single or Multiple}/{date}/{epoch}) [default: None]')
	#parser.add_argument('-generate', action='store_true', default=None, help='Generate pose modified image from given image')

	args = parser.parse_args()

	return args

def main():
	args = parse_args()

	if args.model_type == 'Model_F':
		args.save_dir = os.path.join(args.save_dir, 'Model_F',datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	elif args.model_type == 'Model_P':
		args.save_dir = os.path.join(args.save_dir, 'Model_P',datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	os.makedirs(args.save_dir)

	print("Parameters:")
	for attr, value in sorted(args.__dict__.items()):
		text ="\t{}={}\n".format(attr.upper(), value)
		print(text)
		with open('{}/Parameters.txt'.format(args.save_dir),'a') as f:
			f.write(text)

	if args.model_type == 'Model_F':
		#train model model_F
		print("Do nothing")
	else:
		#Train model_P
		#load front feature from model_F
		front_feat_file= './dataset/cfp-dataset/front_feat.bin'
		profile_feat_file = './dataset/cfp-dataset/profile_feat.bin'
		protocol_dir = './dataset/cfp-dataset/Protocol/Split'
		pair_type = 'FP'
		split = args.split_id
		
		final_front_pair, final_front_id = load_feat(front_feat_file)

		#load image
		images_f, id_labels_f, Nd, channel_num = data_loader(args.data_place, 'Front')
		images_p, id_labels_p, Nd, channel_num = data_loader(args.data_place, 'Profile')

		#make pair data
		final_images_p, final_images_f, id_labels, final_front_feat = make_pair_data(images_p, images_f, id_labels_p, final_front_pair, protocol_dir, pair_type, split)
		
		#initialize model_P
		Nz = 50
		if not args.load_model:
			model_F_D_path = 'C:\\Users\\duyson\\Desktop\\Projects\\FaceNormalize\\PytorchGAN\\snapshot\\Front\\2018-05-10_16-51-09\\epoch1000_D.pt'
			model_F_G_path = 'C:\\Users\\duyson\\Desktop\\Projects\\FaceNormalize\\PytorchGAN\\snapshot\\Front\\2018-05-10_16-51-09\\epoch1000_G.pt'
			
			D_F = Model_F.Discriminator(Nd, channel_num)
			G_F = Model_F.Generator(channel_num)
			D_P = Model_P.Discriminator(Nd,channel_num)
			G_P = Model_P.Generator(Nz, channel_num)

			D_F = torch.load(model_F_D_path)
			G_F = torch.load(model_F_G_path)
			D_P.convLayers = D_F.convLayers
			D_P.fc = D_F.fc
			G_P.G_dec_convLayers_frontal = G_F.G_dec_convLayers
			train_model_P(final_images_p, final_images_f, final_front_feat, id_labels, Nd, Nz, D_P, G_P, args)
		else:
			path_to_G = './snapshot/Model_P/initial_training/epoch1000_G.pt'
			path_to_D = './snapshot/Model_P/initial_training/epoch1000_D.pt'

			D_P = Model_P.Discriminator(Nd,channel_num)
			G_P = Model_P.Generator(Nz, channel_num)

			D_P = torch.load(path_to_D)
			G_P = torch.load(path_to_G)
			train_model_P(final_images_p, final_images_f, final_front_feat, id_labels, Nd, Nz, D_P, G_P, args)

if __name__ == '__main__':
	main()


