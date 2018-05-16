#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
from torch.autograd import Variable
import pdb

def is_D_strong(real_output, syn_output, id_label_tensor, Nd, thresh=0.9):
	
	_, id_real_ans = torch.max(real_output[:, :Nd], 1)
	_, id_syn_ans = torch.max(syn_output[:, :Nd], 1)

	id_real_precision = (id_real_ans==id_label_tensor).type(torch.FloatTensor).sum() / real_output.size()[0]
	gan_real_precision = (real_output[:,Nd].sigmoid()>=0.5).type(torch.FloatTensor).sum() / real_output.size()[0]
	gan_syn_precision = (syn_output[:,Nd].sigmoid()<0.5).type(torch.FloatTensor).sum() / syn_output.size()[0]

	total_precision = (id_real_precision+gan_real_precision+gan_syn_precision)/4

	total_precision = total_precision.data[0]
	if total_precision>=thresh:
		flag_D_strong = True
	else:
		flag_D_strong = False

	return flag_D_strong