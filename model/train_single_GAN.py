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
from utils.is_D_strong import is_D_strong
from utils.log_learning import log_learning
from utils.convert_image import convert_image
from utils.DataAugmentation import FaceIdPoseDataset, Resize, RandomCrop

def train_single_GAN(images, id_labels, Nd, D_model, G_model, args):
	if args.cuda:
		D_model.cuda()
		G_model.cuda()

	D_model.train()
	G_model.train()

	lr_Adam = args.lr
	beta1_Adam = args.beta1
	beta2_Adam = args.beta2

	image_size = images.shape[0]
	epoch_time = np.ceil(image_size / args.batch_size).astype(int)

	optimizer_D = optim.Adam(D_model.parameters(), lr=lr_Adam, betas=(beta1_Adam, beta2_Adam))
	optimizer_G = optim.Adam(G_model.parameters(), lr=lr_Adam, betas=(beta1_Adam, beta2_Adam))
	loss_criterion = nn.CrossEntropyLoss()
	loss_criterion_gan = nn.BCEWithLogitsLoss()

	loss_log = []
	steps = 0

	flag_D_strong = False

	for epoch in range(1, args.epochs + 1):
		
		#load augmented data
		transformed_dataset = FaceIdPoseDataset(images, id_labels,
			transforms = transforms.Compose([
				Resize((110, 110)),
				RandomCrop((96,96))
				]))
		dataloader = DataLoader(transformed_dataset, batch_size = args.batch_size, shuffle=True)

		for i, batch_data in enumerate(dataloader):
			D_model.zero_grad()
			G_model.zero_grad()

			batch_image = torch.FloatTensor(batch_data[0].float())
			batch_id_label = batch_data[1]
			minibatch_size = len(batch_image)

			batch_ones_label = torch.ones(minibatch_size)
			batch_zeros_label = torch.zeros(minibatch_size)

			if args.cuda:
				batch_image, batch_id_label, batch_ones_label, batch_zeros_label = \
					batch_image.cuda(), batch_id_label.cuda(), batch_ones_label.cuda(), batch_zeros_label.cuda()

			batch_image, batch_id_label, batch_ones_label, batch_zeros_label = \
				Variable(batch_image), Variable(batch_id_label), Variable(batch_ones_label), Variable(batch_zeros_label)

			generated = G_model(batch_image)

			steps += 1

			if flag_D_strong:
				
				if i%5 == 0:
					#update D
					flag_D_strong = Learn_D(D_model, loss_criterion, loss_criterion_gan, optimizer_D,
						batch_image, generated, batch_id_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args)

				else:
					#Update G
					Learn_G(D_model, loss_criterion, loss_criterion_gan, optimizer_G, generated,
						batch_id_label, batch_ones_label, epoch, steps, Nd, args)

			else:

				if i%2 == 0:
					#update D
					flag_D_strong = Learn_D(D_model, loss_criterion, loss_criterion_gan, optimizer_D,
						batch_image, generated, batch_id_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args)

				else:
					#Update G
					Learn_G(D_model, loss_criterion, loss_criterion_gan, optimizer_G, generated,
						batch_id_label, batch_ones_label, epoch, steps, Nd, args)

		if epoch%args.save_freq == 0:
			if not os.path.isdir(args.save_dir):
				os.makedirs(args.save_dir)

			save_path_D = os.path.join(args.save_dir, 'epoch{}_D.pt'.format(epoch))
			torch.save(D_model, save_path_D)
			save_path_G = os.path.join(args.save_dir, 'epoch{}_G.pt'.format(epoch))
			torch.save(G_model, save_path_G)

			#save gen image
			save_generated_image = convert_image(generated[0].cpu().data.numpy())
			save_path_image = os.path.join(args.save_dir, 'epoch{}_generatedimage.jpg'.format(epoch))
			misc.imsave(save_path_image, save_generated_image.astype(np.uint8))

def Learn_D(D_model, loss_criterion, loss_criterion_gan, optimizer_D, batch_image, \
			generated, batch_id_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args):
	
	real_output = D_model(batch_image)
	syn_output = D_model(generated.detach())

	L_id = loss_criterion(real_output[:, :Nd], batch_id_label)
	L_gan = loss_criterion_gan(real_output[:, Nd], batch_ones_label) + loss_criterion_gan(syn_output[:, Nd], batch_zeros_label)

	d_loss = L_id + L_gan
	d_loss.backward()
	optimizer_D.step()
	log_learning(epoch, steps, 'D', d_loss.data[0], args)

	#check if Discriminator is strong
	flag_D_strong = is_D_strong(real_output, syn_output, batch_id_label, Nd)

	return flag_D_strong

def Learn_G(D_model, loss_criterion, loss_criterion_gan, optimizer_G ,generated, \
			batch_id_label, batch_ones_label, epoch, steps, Nd, args):
	syn_output = D_model(generated)

	L_id = loss_criterion(syn_output[:, :Nd], batch_id_label)
	L_gan = loss_criterion_gan(syn_output[:, Nd], batch_ones_label)

	g_loss = L_id + L_gan
	g_loss.backward()
	optimizer_G.step()
	log_learning(epoch, steps, 'G', g_loss.data[0], args)
