#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
from scipy import misc
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.log_learning import log_learning
from utils.convert_image import convert_image
from utils.DataAugmentation import FaceIdPoseDataset, Resize, RandomCrop, PFFeatDataset

def train_transfer_block(front_feats, profile_feats, id_labels, Nd, D_model, G_model, args):
	if args.cuda:
		D_model.cuda()
		G_model.cuda()

	D_model.train()
	G_model.train()

	lr_Adam = args.lr
	beta1_Adam = args.beta1
	beta2_Adam = args.beta2

	feat_size = front_feats.shape[0]
	print(feat_size)
	epoch_time = np.ceil(feat_size / args.batch_size).astype(int)

	optimizer_D = optim.Adam(D_model.parameters(), lr=lr_Adam, betas=(beta1_Adam, beta2_Adam))
	optimizer_G = optim.Adam(G_model.parameters(), lr=lr_Adam, betas=(beta1_Adam, beta2_Adam))
	
	loss_criterion = nn.MSELoss()
	loss_criterion_gan = nn.BCEWithLogitsLoss()

	loss_log = []
	steps = 0

	for epoch in range(1, args.epochs + 1):
		transformed_dataset = PFFeatDataset(front_feats, profile_feats, id_labels)
		print(transformed_dataset)
		dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=True)

		for i,batch_data in enumerate(dataloader):
			D_model.zero_grad()
			G_model.zero_grad()

			batch_front_data = torch.FloatTensor(batch_data[0].float())
			batch_profile_data = torch.FloatTensor(batch_data[1].float())
			batch_id_label = batch_data[2]
			minibatch_size = len(batch_front_data)

			batch_ones_label = torch.ones(minibatch_size,1)
			batch_zeros_label = torch.zeros(minibatch_size,1)

			if args.cuda:
				batch_front_data, batch_profile_data, batch_id_label, batch_ones_label, batch_zeros_label = \
				batch_front_data.cuda(), batch_profile_data.cuda(), batch_id_label.cuda(), batch_ones_label.cuda(), batch_zeros_label.cuda()

			batch_front_data, batch_profile_data, batch_id_label, batch_ones_label, batch_zeros_label = \
			Variable(batch_front_data), Variable(batch_profile_data), Variable(batch_id_label), Variable(batch_ones_label), Variable(batch_zeros_label)

			generated = G_model(batch_profile_data)

			steps += 1

			Learn_D(D_model, loss_criterion_gan, optimizer_D, batch_front_data, generated,
				batch_id_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args)

			Learn_G(D_model, loss_criterion, loss_criterion_gan, optimizer_G, batch_front_data, generated, batch_id_label,
				batch_ones_label, epoch, steps, Nd, args)

		if epoch%args.save_freq == 0:
			if not os.path.isdir(args.save_dir):
				os.makedirs(args.save_dir)

			save_path_D = os.path.join(args.save_dir,'epoch{}_D.pt'.format(epoch))
			torch.save(D_model, save_path_D)
			save_path_G = os.path.join(args.save_dir, 'epoch{}_G.pt'.format(epoch))
			torch.save(G_model, save_path_G)

def Learn_D(D_model, loss_criterion_gan, optimizer_D, batch_front_data, generated,
	batch_id_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args):
	real_output = D_model(batch_front_data)
	syn_output = D_model(generated.detach())

	L_gan = loss_criterion_gan(real_output, batch_ones_label) + \
				loss_criterion_gan(syn_output, batch_zeros_label)

	d_loss = L_gan #+ L_id
	d_loss.backward()
	optimizer_D.step()
	log_learning(epoch, steps, 'D', d_loss.data[0], args)

def Learn_G(D_model, loss_criterion, loss_criterion_gan, optimizer_G, batch_front_data, generated, batch_id_label,
	batch_ones_label, epoch, steps, Nd, args):
	syn_output = D_model(generated)
	#L_sim = loss_criterion(generated, batch_front_data)
	L_gan = loss_criterion_gan(syn_output, batch_ones_label)

	g_loss =  L_gan #+ L_sim
	g_loss.backward()
	optimizer_G.step()
	log_learning(epoch, steps, 'G', g_loss.data[0], args)