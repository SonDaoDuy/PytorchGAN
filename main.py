#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import datetime
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from model import single_GAN as single_model
from model import transfer_GAN as transfer_block
from model.train_single_GAN import train_single_GAN
from model.train_transfer_block import train_transfer_block
from utils.data_loader import data_loader
import pdb
from utils.read_bin_cfp import read_bin_cfp


def parse_args():
	parser = argparse.ArgumentParser(description='PF_GAN')
	# learning & saving parameterss
	parser.add_argument('-lr', type=float, default=0.0002, help='initial learning rate [default: 0.0002]')
	parser.add_argument('-beta1', type=float, default=0.5, help='adam optimizer parameter [default: 0.5]')
	parser.add_argument('-beta2', type=float, default=0.999, help='adam optimizer parameter [default: 0.999]')
	parser.add_argument('-epochs', type=int, default=1000, help='number of epochs for train [default: 1000]')
	parser.add_argument('-batch_size', type=int, default=64, help='batch size for training [default: 8]')
	parser.add_argument('-save_dir', type=str, default='snapshot', help='where to save the snapshot')
	parser.add_argument('-save_freq', type=int, default=1, help='save learned model for every "-save-freq" epoch')
	parser.add_argument('-cuda', action='store_true', default=True, help='enable the gpu')
	# data souce
	#parser.add_argument('-random', action='store_true', default=False, help='use randomely created data to run program')
	parser.add_argument('-data_place', type=str, default='./dataset', help='prepared data path to run program')
	# model
	parser.add_argument('-model_type', type=str, default='Front', help='use multi image DR_GAN model')
	#parser.add_argument('-images-perID', type=int, default=0, help='number of images per person to input to multi image DR_GAN')
	# option
	#parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot(snapshot/{Single or Multiple}/{date}/{epoch}) [default: None]')
	#parser.add_argument('-generate', action='store_true', default=None, help='Generate pose modified image from given image')

	args = parser.parse_args()

	return args

def main():
	args = parse_args()

	if args.model_type == 'Front':
		args.save_dir = os.path.join(args.save_dir, 'Front',datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	elif args.model_type == 'Profile':
		args.save_dir = os.path.join(args.save_dir, 'Profile',datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	else:
		args.save_dir = os.path.join(args.save_dir, 'Transfer',datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	os.makedirs(args.save_dir)

	print("Parameters:")
	for attr, value in sorted(args.__dict__.items()):
		text ="\t{}={}\n".format(attr.upper(), value)
		print(text)
		with open('{}/Parameters.txt'.format(args.save_dir),'a') as f:
			f.write(text)

	if args.model_type != 'Transfer':
		#input data
		images, id_labels, Nd, channel_num = data_loader(args.data_place, args.model_type)
		D = single_model.Discriminator(Nd, channel_num)
		G = single_model.Generator(channel_num)
		train_single_GAN(images, id_labels, Nd, D, G, args)
	else:
		#initialize G and D for transfer block
		front_feat_file= './dataset/cfp-dataset/front_feat.bin'
		profile_feat_file = './dataset/cfp-dataset/profile_feat.bin'
		protocol_dir = './dataset/cfp-dataset/Protocol/Split'
		pair_type = 'FP'
		
		final_front_pair, final_profile_pair, final_front_id, final_profile_id = \
		read_bin_cfp(front_feat_file, profile_feat_file, protocol_dir, pair_type)

		# print(np.shape(final_front_pair), np.shape(final_profile_pair))
		# print(np.count_nonzero(final_front_id - final_profile_id))
		Nd = 500
		D = transfer_block.Discriminator(320,1)
		#D = transfer_block.Discriminator(320,Nd + 1)
		G = transfer_block.Generator(320, 320)
		train_transfer_block(final_front_pair, final_profile_pair, final_front_id, Nd, D, G, args)

if __name__ == '__main__':
	main()


