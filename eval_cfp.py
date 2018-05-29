import os
import argparse
import datetime
import numpy as np
import torch
import struct as st
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from model.version_2 import model_P as Model_P
from model import transfer_GAN as transfer_block
from utils.data_loader import data_loader_ijba, data_loader
from utils.DataAugmentation import FaceIdPoseDataset, Resize, RandomCrop

from evals.eval_roc_cfp import eval_roc_main

import pdb

def parse_args():
	parser = argparse.ArgumentParser(description='PF_GAN')
	# learning & saving parameterss
	parser.add_argument('-batch_size', type=int, default=64, help='batch size for training [default: 8]')
	parser.add_argument('-cuda', action='store_true', default=True, help='enable the gpu')
	parser.add_argument('-data_name', type=str, default='cfp', help='Name of evaluation dataset')
	parser.add_argument('-data_place', type=str, default='./dataset', help='prepared data path to run program')
	parser.add_argument('-model_dir', type=str, default='./snapshot/Model_P', help='use multi image DR_GAN model')
	parser.add_argument('-save_dir', type=str, default='./results/test_on_cfp', help='test result folder')

	args = parser.parse_args()

	return args

def extract_feat(G_model, D_model, frontal_img, id_label_f, profile_img, id_label_p, data_num_f, data_num_p, split_id, args):
	if args.cuda:
		G_model.cuda()
		D_model.cuda()

	G_model.eval()
	D_model.eval()

	frontal_dataset = FaceIdPoseDataset(frontal_img, id_label_f, 
		transforms = transforms.Compose([
			Resize((110, 110)),
			RandomCrop((96,96))
			]))
	frontal_loader = DataLoader(frontal_dataset, 
		batch_size=args.batch_size, shuffle=False,
		pin_memory=True)
	profile_dataset = FaceIdPoseDataset(profile_img, id_label_p, 
		transforms = transforms.Compose([
			Resize((110, 110)),
			RandomCrop((96,96))
			]))
	profile_loader = DataLoader(profile_dataset, 
		batch_size=args.batch_size, shuffle=False,
		pin_memory=True)

	feat_dim = 320
	
	frontal_feat_file = os.path.join(args.save_dir, 'frontal_feat_' + split_id +'.bin')
	with open(frontal_feat_file, 'wb') as bin_f:
		bin_f.write(st.pack('ii', data_num_f, feat_dim))
		for i, batch_data in enumerate(frontal_loader):
			batch_front_img = torch.FloatTensor(batch_data[0].float())
			#batch_id_label = batch_data[1]
			minibatch_size = len(batch_front_img)
			fixed_noise = torch.FloatTensor(np.random.uniform(-1,1, (minibatch_size, 50)))
			if args.cuda:
				batch_front_img = batch_front_img.cuda()
				fixed_noise = fixed_noise.cuda()

			batch_front_img, fixed_noise = Variable(batch_front_img), Variable(fixed_noise)

			_, gen_img, _ = G_model(batch_front_img, fixed_noise)
			output = D_model(gen_img)
			features = D_model.features
			output_data = features.cpu().data.numpy()
			output_size = features.size(0)

			for j in range(output_size):
				bin_f.write(st.pack('f'*feat_dim, *tuple(output_data[j, :])))

	print("we have done frontal split " + split_id)

	profile_feat_file = os.path.join(args.save_dir, 'profile_feat_' + split_id + '.bin')
	with open(profile_feat_file, 'wb') as bin_f:
		bin_f.write(st.pack('ii', data_num_p, feat_dim))
		for i, batch_data in enumerate(profile_loader):
			batch_profile_img = torch.FloatTensor(batch_data[0].float())
			#batch_id_label = batch_data[1]
			minibatch_size = len(batch_profile_img)
			fixed_noise = torch.FloatTensor(np.random.uniform(-1,1, (minibatch_size, 50)))
			if args.cuda:
				batch_profile_img = batch_profile_img.cuda()
				fixed_noise = fixed_noise.cuda()

			batch_profile_img, fixed_noise = Variable(batch_profile_img), Variable(fixed_noise)

			# _, gen_img, _ = G_model(batch_profile_img, fixed_noise)
			# output = D_model(gen_img)
			# features = D_model.features
			# output_data = features.cpu().data.numpy()
			# output_size = features.size(0)
			_, _, features = G_model(batch_profile_img, fixed_noise)
			output_data = features.cpu().data.numpy()
			output_size = features.size(0)
			for j in range(output_size):
				bin_f.write(st.pack('f'*feat_dim, *tuple(output_data[j, :])))
	
	print("we have done profile split "+ split_id)



def main():
	args = parse_args()

	infos = [
	('cfp_split_10/epoch1000_G.pt', 'cfp_split_10/epoch1000_D.pt', '10'),
	('cfp_split_09/epoch1000_G.pt', 'cfp_split_09/epoch1000_D.pt', '09'),
	('cfp_split_08/epoch1000_G.pt', 'cfp_split_08/epoch1000_D.pt', '08'),
	('cfp_split_07/epoch729_G.pt', 'cfp_split_07/epoch729_D.pt', '07'),
	('cfp_split_06/epoch1000_G.pt', 'cfp_split_06/epoch1000_D.pt', '06'),
	('cfp_split_05/epoch1000_G.pt', 'cfp_split_05/epoch1000_D.pt', '05'),
	('cfp_split_04/epoch1000_G.pt', 'cfp_split_04/epoch1000_D.pt', '04'),
	('cfp_split_03/epoch1000_G.pt', 'cfp_split_03/epoch1000_D.pt', '03'),
	('cfp_split_02/epoch1000_G.pt', 'cfp_split_02/epoch1000_D.pt', '02'),
	('cfp_split_01/epoch1000_G.pt', 'cfp_split_01/epoch1000_D.pt', '01')
	]
	model_name = 'CFP'
	args.save_dir = os.path.join(args.save_dir, 'Evaluate', model_name)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
	G_model = Model_P.Generator(50, 3)
	D_model = Model_P.Discriminator(500, 3)
	frontal_img, id_label_f, Nd, channel_num = data_loader(args.data_place, 'Front')
	profile_img, id_label_p, Nd, channel_num = data_loader(args.data_place, 'Profile')
	data_num_f = len(frontal_img)
	data_num_p = len(profile_img)

	for info in infos:
		G_model_path, D_model_path, split_id = info
		G_model_path = os.path.join(args.model_dir, G_model_path)
		D_model_path = os.path.join(args.model_dir, D_model_path)
		G_model = torch.load(G_model_path)
		D_model = torch.load(D_model_path)
		#extract_feat(G_model, D_model, frontal_img, id_label_f, profile_img, id_label_p, data_num_f, data_num_p, split_id, args)
	
	eval_roc_main(args.save_dir, args.save_dir)

if __name__ == '__main__':
	main()