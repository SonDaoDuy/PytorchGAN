import argparse
import os,sys,shutil
import time
import struct as st
from utils.data_loader import data_loader
from utils.DataAugmentation import FaceIdPoseDataset, Resize, RandomCrop
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
from torch.autograd import Variable
from model import single_GAN as single_model
from torch.utils.data import DataLoader
import pdb

def parse_args():
	parser = argparse.ArgumentParser(description='Save feature to .bin files after train model')

	#parser.add_argument('-image_type', type=str, default='Front', help='Type of image (front or profile)')
	parser.add_argument('-save_dir', type=str, default='./dataset/cfp-dataset', help='location to store .bin feature')
	parser.add_argument('-data_dir', type=str, default='./dataset/cfp-dataset', help='location of the dataset')
	parser.add_argument('-cuda', action='store_true', default=True, help='enable gpu')
	parser.add_argument('-model_dir', type=str, default='./snapshot', help='model parameter directory')
	parser.add_argument('-batch_size', type=int, default=64, help='batch size for loading data')
	args = parser.parse_args()

	return args

def main():
	args = parse_args()


	front_feat_dir = os.path.join(args.save_dir, 'front_feat.bin')
	model_G_path_f = os.path.join(args.model_dir, 'Front', '2018-05-10_16-51-09', 'epoch1000_G.pt')
	#model_D_path = os.path.join(args.model_dir, 'Front', '2018-05-10_16-51-09', 'epoch1000_D.pt')

	profile_feat_dir = os.path.join(args.save_dir, 'profile_feat.bin')
	model_G_path_p = os.path.join(args.model_dir, 'Profile', '2018-05-11_09-09-05', 'epoch1000_G.pt')
	#model_D_path = os.path.join(args.model_dir, 'Front', '2018-05-10_16-51-09', 'epoch1000_D.pt')

	#prepare input image
	images_f, id_labels_f, Nd, channel_num = data_loader(args.data_dir, 'Front')
	images_p, id_labels_p, Nd, channel_num = data_loader(args.data_dir, 'Profile')

	image_data_f = FaceIdPoseDataset(images_f, id_labels_f,
			transforms=transforms.Compose([
				Resize((110, 110)),
				RandomCrop((96,96))
				]))
	dataloader_f = DataLoader(image_data_f, batch_size=args.batch_size, shuffle=False, pin_memory=True)

	image_data_p = FaceIdPoseDataset(images_p, id_labels_p,
			transforms=transforms.Compose([
				Resize((110, 110)),
				RandomCrop((96,96))
				]))
	dataloader_p = DataLoader(image_data_p, batch_size=args.batch_size, shuffle=False, pin_memory=True)

	#prepare model
	G_f = single_model.Generator(channel_num)
	G_p = single_model.Generator(channel_num)

	G_f = torch.load(model_G_path_f)
	G_p = torch.load(model_G_path_p)


	data_num_f = len(images_f)
	data_num_p = len(images_p)
	feat_dim = 320
	id_label = 1

	if args.cuda:
		G_f = G_f.cuda()
		G_p = G_p.cuda()

	G_f.eval()
	G_p.eval()
	#D.eval()


	with open(front_feat_dir, 'wb') as bin_f:
		bin_f.write(st.pack('iii', data_num_f, feat_dim, id_label))
		for i, batch_data in enumerate(dataloader_f):
			features = []
			batch_image = torch.FloatTensor(batch_data[0].float())
			batch_id_label = batch_data[1]

			if args.cuda:
				batch_image = batch_image.cuda()

			batch_image = Variable(batch_image)
			generated = G_f(batch_image)
			out_feat = G_f.features.cpu().data.numpy() #get feature vectors
			print(batch_id_label)
			feat_num = G_f.features.size(0)

			for j in range(feat_num):
				bin_f.write(st.pack('f'*feat_dim + 'i', *tuple(out_feat[j, :]), batch_id_label[j]))

	with open(profile_feat_dir, 'wb') as bin_f:
		bin_f.write(st.pack('iii', data_num_p, feat_dim, id_label))
		for i, batch_data in enumerate(dataloader_p):
			features = []
			batch_image = torch.FloatTensor(batch_data[0].float())
			batch_id_label = batch_data[1]

			if args.cuda:
				batch_image = batch_image.cuda()

			batch_image = Variable(batch_image)
			generated = G_p(batch_image)
			out_feat = G_p.features.cpu().data.numpy() #get feature vectors
			print(batch_id_label)
			feat_num = G_p.features.size(0)

			for j in range(feat_num):
				bin_f.write(st.pack('f'*feat_dim + 'i', *tuple(out_feat[j, :]), batch_id_label[j]))

if __name__ == '__main__':
	main()







