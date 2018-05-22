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
from utils.data_loader import data_loader_ijba
from utils.DataAugmentation import IJBADataset, Resize, RandomCrop

from evals.test_verify_ijba import test_verify 
from evals.test_recog_ijba import test_recog

import pdb

def parse_args():
	parser = argparse.ArgumentParser(description='PF_GAN')
	# learning & saving parameterss
	parser.add_argument('-batch_size', type=int, default=64, help='batch size for training [default: 8]')
	parser.add_argument('-cuda', action='store_true', default=True, help='enable the gpu')
	parser.add_argument('-data_name', type=str, default='ijba', help='Name of evaluation dataset')
	parser.add_argument('-data_place', type=str, default='./dataset', help='prepared data path to run program')
	parser.add_argument('-model_dir', type=str, default='./snapshot', help='use multi image DR_GAN model')

	args = parser.parse_args()

	return args

def extract_feat(G_model, D_model, yaw_type, args):

	if args.cuda:
		G_model.cuda()
		D_model.cuda()

	G_model.eval()
	D_model.eval()

	infos = [ ('C:\\Users\\duyson\\Desktop\\Projects\\FaceNormalize\\PytorchGAN\\dataset\\IJBA\\IJBA\\align_image_11', 'ijb_a_11_align_split', 'frame'),
			('C:\\Users\\duyson\\Desktop\\Projects\\FaceNormalize\\PytorchGAN\\dataset\\IJBA\\IJBA\\align_image_11', 'ijb_a_11_align_split', 'img'), 
			('C:\\Users\\duyson\\Desktop\\Projects\\FaceNormalize\\PytorchGAN\\dataset\\IJBA\\IJBA\\align_image_1N', 'split', 'gallery'),
			('C:\\Users\\duyson\\Desktop\\Projects\\FaceNormalize\\PytorchGAN\\dataset\\IJBA\\IJBA\\align_image_1N', 'split', 'probe') ] 

	for root_dir, sub_dir, img_type in infos:
		for split in range(1, 11):
			split_dir = os.path.join(root_dir, sub_dir + str(split))
			img_dir = os.path.join(split_dir, img_type)
			img_list_file = os.path.join(split_dir, '{}_list_{}.txt'.format(img_type, yaw_type))

			images_ijba, channel_num = data_loader_ijba(img_dir, img_list_file)

			img_dataset = IJBADataset(images_ijba,
				transforms=transforms.Compose([
				Resize((110, 110)),
				RandomCrop((96,96))
				]))

			dataloader = DataLoader(img_dataset, batch_size = args.batch_size,
				shuffle = False,
				pin_memory = True)

			data_num = len(img_dataset)
			img_feat_file = os.path.join(split_dir, '{}_feat.bin'.format(img_type))
			feat_dim = 320
			with open(img_feat_file, 'wb') as bin_f:
				bin_f.write(st.pack('ii', data_num, feat_dim))
				for i, input_img in enumerate(dataloader):
					input_img = torch.FloatTensor(input_img.float())
					fixed_noise = torch.FloatTensor(np.random.uniform(-1,1, (input_img.size(0), 50)))
					if args.cuda:
						input_img = input_img.cuda()
						fixed_noise = fixed_noise.cuda()

					input_img = Variable(input_img)
					fixed_noise = Variable(fixed_noise)
					#get output from model
					# _, generated, _ = G_model(input_img, fixed_noise)
					# output = D_model(generated)
					# features = D_model.features
					# output_data = features.cpu().data.numpy()
					# output_size = features.size(0)
					_, _, features = G_model(input_img, fixed_noise)
					output_data = features.cpu().data.numpy()
					output_size = features.size(0)
					# save feat to bin file
					for j in range(output_size):
						bin_f.write(st.pack('f'*feat_dim, *tuple(output_data[j,:])))

			print('we have complete {} {}'.format(img_type, split))

def main():
	args = parse_args()
	
	infos = [
	'./snapshot/Model_P/2018-05-21_15-43-14/epoch1000_G.pt',
	'./snapshot/Model_P/2018-05-21_15-43-14/epoch1000_D.pt',
	'nonli'
	]

	G_model_path, D_model_path, yaw_type = infos
	G_model = Model_P.Generator(50,3)
	D_model = Model_P.Discriminator(500,3)
	G_model = torch.load(G_model_path)
	D_model = torch.load(D_model_path)
	extract_feat(G_model, D_model, yaw_type, args)

	test_recog()
	test_verify()

if __name__ == '__main__':
	main()