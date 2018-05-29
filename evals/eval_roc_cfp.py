import os, sys, shutil
import numpy as np
import matplotlib.pyplot as plt
import math
import random as rd
import struct as st
from scipy import spatial
from sklearn import metrics
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import cosine_similarity
from utils.log_result import log_result

def draw_roc(fpr, tpr, save_dir, split_name, pair_type):
	title = 'center_loss'
	img_name="roc_" + split_name + '_' + pair_type +".png"
	save_dir = os.path.join(save_dir, img_name)
	plt.figure(figsize=(16, 8))
	plt.yticks(np.arange(0.0, 1.05, 0.05))
	plt.xticks(np.arange(0.0, 1.05, 0.05))
	plt.title(title)
	plt.plot(fpr, tpr, linewidth=1, color='r')
	#plt.xscale('log')
	#plt.yscale('log')
	plt.savefig(save_dir)

def load_feat(feat_file):
	feats = list()
	with open(feat_file, 'rb') as in_f:
		feat_num, feat_dim = st.unpack('ii', in_f.read(8))
		for i in range(feat_num):
			feat = np.array(st.unpack('f'*feat_dim, in_f.read(4*feat_dim)))
			feats.append(feat)
	return feats

def calc_eer(fpr, tpr, method=0):
	if method == 0:
		min_dis, eer = 100.0, 1.0
		for i in range(fpr.size):
			if(fpr[i]+tpr[i] > 1.0):
				break
			mid_res = abs(fpr[i]+tpr[i]-1.0)
			if(mid_res < min_dis):
				min_dis = mid_res
				eer = fpr[i]
		return eer
	else:
		f = lambda x: np.interp(x, fpr, tpr)+x-1
		return fsolve(f, 0.0)

def eval_roc(save_dir, protocol_dir, pair_type, split_name, frontal_feats, profile_feats):
	labels, scores = [],[]
	for idx, pair_file in enumerate(['diff.txt', 'same.txt']):
		label = idx
		full_pair_file = protocol_dir+'/'+pair_type+'/'+split_name+'/'+pair_file
		with open(full_pair_file, 'r') as in_f:
			for line in in_f:
				record = line.strip().split(',')
				pair1, pair2 = int(record[0]),int(record[1])
				if pair_type == 'FF':
					vec1, vec2 = frontal_feats[pair1-1], frontal_feats[pair2-1]
				else:
					vec1, vec2 = frontal_feats[pair1-1], profile_feats[pair2-1]
				score = 1-spatial.distance.cosine(vec1,vec2)
				scores.append(score)
				labels.append(label)
	fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
	draw_roc(fpr,tpr, save_dir, split_name, pair_type)
	auc = metrics.auc(fpr, tpr)
	eer = calc_eer(fpr, tpr)
	return auc, eer

def eval_roc_main(save_dir, bin_dir):
	front_feat_file = 'frontal_feat_'
	profile_feat_file = 'profile_feat_'

	protocol_dir = './dataset/cfp-dataset/Protocol/Split'
	pair_types = ['FF', 'FP']
	split_num = 10

	for pair_type in pair_types:
		print_info = 'Frontal-Frontal' if pair_type=='FF' else 'Frontal-Profile'
		text = '----- result for' + print_info + '-----'
		print(text)
		log_result(text, save_dir)
		aucs, eers = list(), list()
		for split_id in range(split_num):
			split_name = str(split_id+1)
			if len(split_name) < 2:
				split_name = '0' + split_name
			split_front_feat = os.path.join(bin_dir, front_feat_file + split_name + '.bin')
			print(split_front_feat)
			split_profile_feat = os.path.join(bin_dir, profile_feat_file + split_name + '.bin')
			frontal_feats = load_feat(split_front_feat)
			profile_feats = load_feat(split_profile_feat)
			auc, eer = eval_roc(save_dir, protocol_dir, pair_type, split_name, frontal_feats, profile_feats)

			aucs.append(auc)
			eers.append(eer)
		print(aucs)
		text = 'Average auc:' + str(np.mean(aucs))
		print(text)
		log_result(text, save_dir)
		print(eers)
		text = 'Average eer:' + str(np.mean(eers))
		print(text)
		log_result(text, save_dir)
