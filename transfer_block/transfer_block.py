#!/usr/bin/env python
# encoding: utf-8

import torch
from torch import nn, optim
from torch.autograd import Variable
import pdb


class Discriminator(nn.Module):
	"""docstring for Discriminator"""
	def __init__(self, input_dim, output_dim):
		super(Discriminator, self).__init__()

		fc = [
			nn.Linear(input_dim, 512),
			nn.Relu(),
			nn.Linear(512, 256),
			nn.Relu(),
			nn.Linear(256, output_dim),
			nn.Sigmoid(),
		]

		self.fc = nn.Sequential(*fc)

		for m in nn.modules():
			if isinstance(nn.Linear):
				m.weight.data.normal_(0, 0.02)

	def forward(self, input):
		x = self.fc(input)
		return x

class Generator(nn.Module):
	"""docstring for Generator"""
	def __init__(self, input_dim, output_dim):
		super(Generator, self).__init__()

		lnLayers = [
			nn.Linear(input_dim, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Linear(256, 512),
			nn.BatchNorm1d(512),
			nn.Relu(),
			nn.Linear(512, 1024),
			nn.BatchNorm1d(1024),
			nn.Relu(),
			nn.Linear(1024, output_dim),
			nn.Tanh(),
		]

		self.lnLayers = nn.Sequential(*lnLayers)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.02)

	def forward(self, input):
		x = self.lnLayers(input)
		return x
		
