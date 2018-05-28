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
from utils.DataAugmentation import FaceIdPoseDataset_v2, Resize, RandomCrop
from matplotlib import pylab

def train_model_P(images_profile, images_front, frontal_feats, id_labels, Nd, Nz, D_model, G_model, args):
	if args.cuda:
		D_model.cuda()
		G_model.cuda()

	D_model.train()
	G_model.train()

	lr_Adam = args.lr
	beta1_Adam = args.beta1
	beta2_Adam = args.beta2

	image_size = images_profile.shape[0]
	epoch_time = np.ceil(image_size / args.batch_size).astype(int)

	optimizer_D = optim.Adam(D_model.parameters(), lr=lr_Adam, betas=(beta1_Adam, beta2_Adam))
	optimizer_G = optim.Adam(G_model.parameters(), lr=lr_Adam, betas=(beta1_Adam, beta2_Adam))
	
	loss_criterion_emb = nn.CosineEmbeddingLoss()
	loss_criterion_sim = nn.MSELoss()
	#loss_criterion_sim = nn.L1Loss()
	loss_criterion_id = nn.CrossEntropyLoss()
	loss_criterion_gan = nn.BCEWithLogitsLoss()

	loss_log = []
	steps = 0

	for epoch in range(1, args.epochs + 1):
		
		#load augmented data
		transformed_dataset = FaceIdPoseDataset_v2(images_profile, images_front, frontal_feats, id_labels,
			transforms = transforms.Compose([
				Resize((110, 110)),
				RandomCrop((96,96))
				]))
		dataloader = DataLoader(transformed_dataset, batch_size = args.batch_size, shuffle=True)

		for i, batch_data in enumerate(dataloader):
			D_model.zero_grad()
			G_model.zero_grad()

			batch_image_profile = torch.FloatTensor(batch_data[0].float())
			batch_image_front = torch.FloatTensor(batch_data[1].float())
			batch_front_feats = torch.FloatTensor(batch_data[2].float())
			batch_id_label = batch_data[3]
			minibatch_size = len(batch_image_profile)

			batch_ones_label = torch.ones(minibatch_size)
			batch_zeros_label = torch.zeros(minibatch_size)

			fixed_noise = torch.FloatTensor(np.random.uniform(-1,1, (minibatch_size, Nz)))

			if args.cuda:
				batch_image_profile, batch_image_front, batch_front_feats, batch_id_label, batch_ones_label, batch_zeros_label = \
					batch_image_profile.cuda(), batch_image_front.cuda(), batch_front_feats.cuda(), batch_id_label.cuda(), batch_ones_label.cuda(), batch_zeros_label.cuda()

				fixed_noise = fixed_noise.cuda()

			batch_image_profile, batch_image_front, batch_front_feats, batch_id_label, batch_ones_label, batch_zeros_label = \
				Variable(batch_image_profile), Variable(batch_image_front), Variable(batch_front_feats), Variable(batch_id_label).long(), Variable(batch_ones_label), Variable(batch_zeros_label)

			fixed_noise = Variable(fixed_noise)

			gen_profile_img, gen_frontal_img, gen_trans_feat = G_model(batch_image_profile, fixed_noise)

			steps += 1

			Learn_D(D_model, loss_criterion_id, loss_criterion_gan, optimizer_D, batch_image_front, gen_frontal_img, \
				batch_id_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args)

			#Learn_G_AE(loss_criterion_sim, optimizer_G, gen_profile_img, batch_image_profile, epoch, steps, Nd, args)

			Learn_G(D_model, loss_criterion_emb, loss_criterion_sim, loss_criterion_id, loss_criterion_gan, optimizer_G , gen_profile_img, \
				gen_frontal_img, gen_trans_feat, batch_image_profile, batch_image_front, batch_front_feats, \
				batch_id_label, batch_ones_label, epoch, steps, Nd, args)

		if epoch%args.save_freq == 0:
			if not os.path.isdir(args.save_dir):
				os.makedirs(args.save_dir)

			save_path_D = os.path.join(args.save_dir, 'epoch{}_D.pt'.format(epoch))
			torch.save(D_model, save_path_D)
			save_path_G = os.path.join(args.save_dir, 'epoch{}_G.pt'.format(epoch))
			torch.save(G_model, save_path_G)

			#save gen image
			save_generated_image = convert_image(gen_frontal_img.cpu().data.numpy())
			save_front_image = convert_image(batch_image_front.cpu().data.numpy())
			save_profile_image = convert_image(batch_image_profile.cpu().data.numpy())
			save_path_gen_image = os.path.join(args.save_dir, 'epoch{}_generatedimage.png'.format(epoch))
			save_path_image = os.path.join(args.save_dir, 'epoch{}_frontimage.png'.format(epoch))
			save_path_profile = os.path.join(args.save_dir, 'epoch{}_profileimage.png'.format(epoch))
			# misc.imsave(save_path_gen_image, save_generated_image.astype(np.uint8))
			# misc.imsave(save_path_image, save_front_image.astype(np.uint8))
			save_image(save_front_image, save_generated_image, save_profile_image, save_path_image, save_path_gen_image, save_path_profile)

def save_image(save_front_image, generated_image, save_profile_image, save_img_path, save_gen_img_path, save_path_profile):
	r, c = 2, 7
	G_imgs = save_front_image
	print(len(G_imgs))
	fig, axs = pylab.subplots(r, c)
	cnt = 0
	for i in range(r):
		for j in range(c):
			axs[i,j].imshow(G_imgs[cnt])
			axs[i,j].axis('off')
			cnt += 1
	fig.savefig(save_img_path)

	fig, axs = pylab.subplots(r, c)
	cnt = 0
	for i in range(r):
		for j in range(c):
			axs[i,j].imshow(save_profile_image[cnt])
			axs[i,j].axis('off')
			cnt += 1
	fig.savefig(save_path_profile)

	gen_imgs = generated_image

	fig, axs = pylab.subplots(r, c)
	cnt = 0
	for i in range(r):
		for j in range(c):
			axs[i,j].imshow(gen_imgs[cnt])
			axs[i,j].axis('off')
			cnt += 1
	fig.savefig(save_gen_img_path)
	pylab.close()

def Learn_D(D_model, loss_criterion_id, loss_criterion_gan, optimizer_D, batch_image_front, gen_frontal_img, \
				batch_id_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args):
	real_output = D_model(batch_image_front)
	syn_output = D_model(gen_frontal_img.detach())

	L_id = loss_criterion_id(real_output[:, :Nd], batch_id_label)
	L_gan = loss_criterion_gan(real_output[:, Nd], batch_ones_label) + loss_criterion_gan(syn_output[:, Nd], batch_zeros_label)

	d_loss = L_gan + L_id
	d_loss.backward()
	optimizer_D.step()
	log_learning(epoch, steps, 'D', d_loss.data[0], args)

def Learn_G(D_model, loss_criterion_emb, loss_criterion_sim, loss_criterion_id, loss_criterion_gan, optimizer_G , gen_profile_img, \
				gen_frontal_img, gen_trans_feat, batch_image_profile, batch_image_front, batch_front_feats, \
				batch_id_label, batch_ones_label, epoch, steps, Nd, args):
	lamda_id = 0.003
	lamda_gan = 0.05
	lamda_sym = 0.3
	syn_output=D_model(gen_frontal_img)
	#make flip image
	flip_gen_images = gen_frontal_img.cpu().data.numpy()
	flip_gen_images = np.fliplr(flip_gen_images)
	flip_gen_images = torch.FloatTensor(flip_gen_images.astype(float))
	if args.cuda:
		flip_gen_images = flip_gen_images.cuda()

	flip_gen_images = Variable(flip_gen_images)

	#calculate loss
	L_sym = loss_criterion_sim(gen_frontal_img, flip_gen_images)
	L_emb = loss_criterion_emb(batch_front_feats, gen_trans_feat, batch_ones_label)
	#L_sim_AE = loss_criterion_sim(batch_image_profile, gen_profile_img)
	L_sim_GAN = loss_criterion_sim(gen_frontal_img, batch_image_front)
	L_id = loss_criterion_id(syn_output[:, :Nd], batch_id_label)
	L_gan = loss_criterion_gan(syn_output[:, Nd], batch_ones_label)

	g_loss = lamda_gan*L_gan + lamda_id*L_id + L_sim_GAN  + lamda_id*L_emb + lamda_sym*L_sym

	g_loss.backward()
	optimizer_G.step()
	log_learning(epoch, steps, 'G', g_loss.data[0], args)


def Learn_G_AE(loss_criterion_sim, optimizer_G, gen_profile_img, batch_image_profile, epoch, steps, Nd, args):
	l_sim_AE = loss_criterion_sim(batch_image_profile, gen_profile_img)

	g_ae_loss = l_sim_AE
	g_ae_loss.backward()
	optimizer_G.step()
	log_learning(epoch, steps, 'G_AE', g_ae_loss.data[0], args)

# def Learn_G_T():
# 	pass