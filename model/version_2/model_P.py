#!/usr/bin/env python
# encoding: utf-8

import torch
from torch import nn, optim
from torch.autograd import Variable

class Discriminator(nn.Module):
	"""docstring for Discriminator"""
	def __init__(self, Nd, channel_num):
		super(Discriminator, self).__init__()
		self.features = []
		convLayers = [
			nn.Conv2d(channel_num, 32, 3, 1, 1, bias=False),
			nn.BatchNorm2d(32),
			nn.ELU(),
			nn.Conv2d(32, 64, 3, 1, 1, bias=False), # Bx32x96x96 -> Bx64x96x96
			nn.BatchNorm2d(64),
			nn.ELU(),
			nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx64x96x96 -> Bx64x97x97
			nn.Conv2d(64, 64, 3, 2, 0, bias=False), # Bx64x97x97 -> Bx64x48x48
			nn.BatchNorm2d(64),
			nn.ELU(),
			nn.Conv2d(64, 64, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx64x48x48
			nn.BatchNorm2d(64),
			nn.ELU(),
			nn.Conv2d(64, 128, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx128x48x48
			nn.BatchNorm2d(128),
			nn.ELU(),
			nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx128x48x48 -> Bx128x49x49
			nn.Conv2d(128, 128, 3, 2, 0, bias=False), #  Bx128x49x49 -> Bx128x24x24
			nn.BatchNorm2d(128),
			nn.ELU(),
			nn.Conv2d(128, 96, 3, 1, 1, bias=False), #  Bx128x24x24 -> Bx96x24x24
			nn.BatchNorm2d(96),
			nn.ELU(),
			nn.Conv2d(96, 192, 3, 1, 1, bias=False), #  Bx96x24x24 -> Bx192x24x24
			nn.BatchNorm2d(192),
			nn.ELU(),
			nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx192x24x24 -> Bx192x25x25
			nn.Conv2d(192, 192, 3, 2, 0, bias=False), # Bx192x25x25 -> Bx192x12x12
			nn.BatchNorm2d(192),
			nn.ELU(),
			nn.Conv2d(192, 128, 3, 1, 1, bias=False), # Bx192x12x12 -> Bx128x12x12
			nn.BatchNorm2d(128),
			nn.ELU(),
			nn.Conv2d(128, 256, 3, 1, 1, bias=False), # Bx128x12x12 -> Bx256x12x12
			nn.BatchNorm2d(256),
			nn.ELU(),
			nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx256x12x12 -> Bx256x13x13
			nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x13x13 -> Bx256x6x6
			nn.BatchNorm2d(256),
			nn.ELU(),
			nn.Conv2d(256, 160, 3, 1, 1, bias=False), # Bx256x6x6 -> Bx160x6x6
			nn.BatchNorm2d(160),
			nn.ELU(),
			nn.Conv2d(160, 320, 3, 1, 1, bias=False), # Bx160x6x6 -> Bx320x6x6
			nn.BatchNorm2d(320),
			nn.ELU(),
			nn.AvgPool2d(6, stride=1), #  Bx320x6x6 -> Bx320x1x1
		]

		self.convLayers = nn.Sequential(*convLayers)
		self.fc = nn.Linear(320, Nd + 1)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.normal_(0, 0.02)
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.02)


	def forward(self, input):
		x = self.convLayers(input)

		x = x.view(-1, 320)

		self.features = x

		x = self.fc(x)

		return x

class Crop(nn.Module):
	"""
	Generator でのアップサンプリング時に， ダウンサンプル時のZeroPad2d と逆の事をするための関数
	論文著者が Tensorflow で padding='SAME' オプションで自動的にパディングしているのを
	ダウンサンプル時にはZeroPad2dで，アップサンプリング時には Crop で実現

	### init
	crop_list : データの上下左右をそれぞれどれくらい削るか指定
	"""

	def __init__(self, crop_list):
		super(Crop, self).__init__()

		# crop_lsit = [crop_top, crop_bottom, crop_left, crop_right]
		self.crop_list = crop_list

	def forward(self, x):
		B,C,H,W = x.size()
		x = x[:,:, self.crop_list[0] : H - self.crop_list[1] , self.crop_list[2] : W - self.crop_list[3]]

		return x

class Generator(nn.Module):
	"""docstring for Generator"""
	def __init__(self, Nz, channel_num):
		super(Generator, self).__init__()
		self.features = []
		#encoder profile face
		G_enc_convLayers = [
				nn.Conv2d(channel_num, 32, 3, 1, 1, bias=False), # Bx3x96x96 -> Bx32x96x96
				nn.BatchNorm2d(32),
				nn.ELU(),
				nn.Conv2d(32, 64, 3, 1, 1, bias=False), # Bx32x96x96 -> Bx64x96x96
				nn.BatchNorm2d(64),
				nn.ELU(),
				nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx64x96x96 -> Bx64x97x97
				nn.Conv2d(64, 64, 3, 2, 0, bias=False), # Bx64x97x97 -> Bx64x48x48
				nn.BatchNorm2d(64),
				nn.ELU(),
				nn.Conv2d(64, 64, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx64x48x48
				nn.BatchNorm2d(64),
				nn.ELU(),
				nn.Conv2d(64, 128, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx128x48x48
				nn.BatchNorm2d(128),
				nn.ELU(),
				nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx128x48x48 -> Bx128x49x49
				nn.Conv2d(128, 128, 3, 2, 0, bias=False), #  Bx128x49x49 -> Bx128x24x24
				nn.BatchNorm2d(128),
				nn.ELU(),
				nn.Conv2d(128, 96, 3, 1, 1, bias=False), #  Bx128x24x24 -> Bx96x24x24
				nn.BatchNorm2d(96),
				nn.ELU(),
				nn.Conv2d(96, 192, 3, 1, 1, bias=False), #  Bx96x24x24 -> Bx192x24x24
				nn.BatchNorm2d(192),
				nn.ELU(),
				nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx192x24x24 -> Bx192x25x25
				nn.Conv2d(192, 192, 3, 2, 0, bias=False), # Bx192x25x25 -> Bx192x12x12
				nn.BatchNorm2d(192),
				nn.ELU(),
				nn.Conv2d(192, 128, 3, 1, 1, bias=False), # Bx192x12x12 -> Bx128x12x12
				nn.BatchNorm2d(128),
				nn.ELU(),
				nn.Conv2d(128, 256, 3, 1, 1, bias=False), # Bx128x12x12 -> Bx256x12x12
				nn.BatchNorm2d(256),
				nn.ELU(),
				nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx256x12x12 -> Bx256x13x13
				nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x13x13 -> Bx256x6x6
				nn.BatchNorm2d(256),
				nn.ELU(),
				nn.Conv2d(256, 160, 3, 1, 1, bias=False), # Bx256x6x6 -> Bx160x6x6
				nn.BatchNorm2d(160),
				nn.ELU(),
				nn.Conv2d(160, 320, 3, 1, 1, bias=False), # Bx160x6x6 -> Bx320x6x6
				nn.BatchNorm2d(320),
				nn.ELU(),
				nn.AvgPool2d(6, stride=1), #  Bx320x6x6 -> Bx320x1x1

		]
		self.G_enc_convLayers = nn.Sequential(*G_enc_convLayers)

		#Decoder profile face
		G_dec_convLayers = [
			nn.ConvTranspose2d(320,160, 3,1,1, bias=False), # Bx320x6x6 -> Bx160x6x6
			nn.BatchNorm2d(160),
			nn.ELU(),
			nn.ConvTranspose2d(160, 256, 3,1,1, bias=False), # Bx160x6x6 -> Bx256x6x6
			nn.BatchNorm2d(256),
			nn.ELU(),
			nn.ConvTranspose2d(256, 256, 3,2,0, bias=False), # Bx256x6x6 -> Bx256x13x13
			nn.BatchNorm2d(256),
			nn.ELU(),
			Crop([0, 1, 0, 1]),
			nn.ConvTranspose2d(256, 128, 3,1,1, bias=False), # Bx256x12x12 -> Bx128x12x12
			nn.BatchNorm2d(128),
			nn.ELU(),
			nn.ConvTranspose2d(128, 192,  3,1,1, bias=False), # Bx128x12x12 -> Bx192x12x12
			nn.BatchNorm2d(192),
			nn.ELU(),
			nn.ConvTranspose2d(192, 192,  3,2,0, bias=False), # Bx128x12x12 -> Bx192x25x25
			nn.BatchNorm2d(192),
			nn.ELU(),
			Crop([0, 1, 0, 1]),
			nn.ConvTranspose2d(192, 96,  3,1,1, bias=False), # Bx192x24x24 -> Bx96x24x24
			nn.BatchNorm2d(96),
			nn.ELU(),
			nn.ConvTranspose2d(96, 128,  3,1,1, bias=False), # Bx96x24x24 -> Bx128x24x24
			nn.BatchNorm2d(128),
			nn.ELU(),
			nn.ConvTranspose2d(128, 128,  3,2,0, bias=False), # Bx128x24x24 -> Bx128x49x49
			nn.BatchNorm2d(128),
			nn.ELU(),
			Crop([0, 1, 0, 1]),
			nn.ConvTranspose2d(128, 64,  3,1,1, bias=False), # Bx128x48x48 -> Bx64x48x48
			nn.BatchNorm2d(64),
			nn.ELU(),
			nn.ConvTranspose2d(64, 64,  3,1,1, bias=False), # Bx64x48x48 -> Bx64x48x48
			nn.BatchNorm2d(64),
			nn.ELU(),
			nn.ConvTranspose2d(64, 64,  3,2,0, bias=False), # Bx64x48x48 -> Bx64x97x97
			nn.BatchNorm2d(64),
			nn.ELU(),
			Crop([0, 1, 0, 1]),
			nn.ConvTranspose2d(64, 32,  3,1,1, bias=False), # Bx64x96x96 -> Bx32x96x96
			nn.BatchNorm2d(32),
			nn.ELU(),
			nn.ConvTranspose2d(32, channel_num,  3,1,1, bias=False), # Bx32x96x96 -> Bxchx96x96
			nn.Tanh(),
		]
		self.G_dec_convLayers = nn.Sequential(*G_dec_convLayers)

		#Decoder frontal face
		G_dec_convLayers_frontal = [
			nn.ConvTranspose2d(320,160, 3,1,1, bias=False), # Bx320x6x6 -> Bx160x6x6
			nn.BatchNorm2d(160),
			nn.ELU(),
			nn.ConvTranspose2d(160, 256, 3,1,1, bias=False), # Bx160x6x6 -> Bx256x6x6
			nn.BatchNorm2d(256),
			nn.ELU(),
			nn.ConvTranspose2d(256, 256, 3,2,0, bias=False), # Bx256x6x6 -> Bx256x13x13
			nn.BatchNorm2d(256),
			nn.ELU(),
			Crop([0, 1, 0, 1]),
			nn.ConvTranspose2d(256, 128, 3,1,1, bias=False), # Bx256x12x12 -> Bx128x12x12
			nn.BatchNorm2d(128),
			nn.ELU(),
			nn.ConvTranspose2d(128, 192,  3,1,1, bias=False), # Bx128x12x12 -> Bx192x12x12
			nn.BatchNorm2d(192),
			nn.ELU(),
			nn.ConvTranspose2d(192, 192,  3,2,0, bias=False), # Bx128x12x12 -> Bx192x25x25
			nn.BatchNorm2d(192),
			nn.ELU(),
			Crop([0, 1, 0, 1]),
			nn.ConvTranspose2d(192, 96,  3,1,1, bias=False), # Bx192x24x24 -> Bx96x24x24
			nn.BatchNorm2d(96),
			nn.ELU(),
			nn.ConvTranspose2d(96, 128,  3,1,1, bias=False), # Bx96x24x24 -> Bx128x24x24
			nn.BatchNorm2d(128),
			nn.ELU(),
			nn.ConvTranspose2d(128, 128,  3,2,0, bias=False), # Bx128x24x24 -> Bx128x49x49
			nn.BatchNorm2d(128),
			nn.ELU(),
			Crop([0, 1, 0, 1]),
			nn.ConvTranspose2d(128, 64,  3,1,1, bias=False), # Bx128x48x48 -> Bx64x48x48
			nn.BatchNorm2d(64),
			nn.ELU(),
			nn.ConvTranspose2d(64, 64,  3,1,1, bias=False), # Bx64x48x48 -> Bx64x48x48
			nn.BatchNorm2d(64),
			nn.ELU(),
			nn.ConvTranspose2d(64, 64,  3,2,0, bias=False), # Bx64x48x48 -> Bx64x97x97
			nn.BatchNorm2d(64),
			nn.ELU(),
			Crop([0, 1, 0, 1]),
			nn.ConvTranspose2d(64, 32,  3,1,1, bias=False), # Bx64x96x96 -> Bx32x96x96
			nn.BatchNorm2d(32),
			nn.ELU(),
			nn.ConvTranspose2d(32, channel_num,  3,1,1, bias=False), # Bx32x96x96 -> Bxchx96x96
			nn.Tanh(),
		]
		self.G_dec_convLayers_frontal = nn.Sequential(*G_dec_convLayers_frontal)

		self.G_dec_fc = nn.Linear(320, 320*6*6)

		self.G_dec_fc_frontal = nn.Linear(320 + Nz, 320*6*6)

		#Transfer block
		G_transfer = [
			nn.Linear(320, 512),
			nn.BatchNorm1d(512),
			nn.PReLU(),
			nn.Linear(512, 320),
			nn.BatchNorm1d(320),
			nn.PReLU(),
		]

		self.G_transfer = nn.Sequential(*G_transfer)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.normal_(0, 0.02)

			elif isinstance(m, nn.ConvTranspose2d):
				m.weight.data.normal_(0, 0.02)

			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.02)

	def forward(self, input, noise):
		x = self.G_enc_convLayers(input)

		x = x.view(-1, 320)
		#for profile autoencoder branch
		x_p = self.G_dec_fc(x)
		x_p = x_p.view(-1, 320, 6, 6)
		x_p = self.G_dec_convLayers(x_p)
		#for profile to frontal branch
		x_f = self.G_transfer(x)
		x_f = x_f.view(-1, 320)
		x_trans = torch.add(x, x_f)
		x_f = torch.cat([x_trans, noise], 1)
		x_f = self.G_dec_fc_frontal(x_f)
		x_f = x_f.view(-1,320,6,6)
		x_f = self.G_dec_convLayers_frontal(x_f)

		return x_p, x_f, x_trans
