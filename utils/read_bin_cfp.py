import argparse
import os,sys,shutil
import time
import struct as st
import numpy as np
import pdb

def read_bin_cfp(front_feat_file, profile_feat_file, protocol_dir, pair_type):
	frontal_feats, frontal_ids = load_feat(front_feat_file)
	profile_feats, profile_ids = load_feat(profile_feat_file)

	split_num = 10
	final_front_pair, final_profile_pair, final_front_id, final_profile_id = \
		[], [], [], []

	for split_id in range(split_num):
		split_name = str(split_id + 1)
		if len(split_name)<2:
			split_name = '0'+split_name
		query_folder = os.path.join(protocol_dir, pair_type, split_name)
		pair_front_feat, pair_profile_feat, pair_front_id, pair_profile_id = \
			load_feat_pair(query_folder, frontal_feats, profile_feats, frontal_ids, profile_ids)

		for j in range(len(pair_front_feat)):
			final_front_pair.append(pair_front_feat[j])
			final_profile_pair.append(pair_profile_feat[j])
			#reshape neu can??/
			final_front_id.append(pair_front_id[j])
			final_profile_id.append(pair_profile_id[j])

	#print(np.shape(final_front_pair))
	final_front_pair, final_profile_pair, final_front_id, final_profile_id = \
	np.array(final_front_pair), np.array(final_profile_pair), np.array(final_front_id), np.array(final_profile_id)


	return final_front_pair, final_profile_pair, final_front_id, final_profile_id

def load_feat(bin_file):
	img_feats = []
	img_ids = []
	with open(bin_file, 'rb') as in_f:
		data_num, feat_dim, label = st.unpack('iii', in_f.read(12))
		for i in range(data_num):
			content = np.array(st.unpack('f'*feat_dim + 'i', in_f.read(4*feat_dim + 4)))
			feat = content[:-1]
			img_id = content[-1].astype(int)
			img_feats.append(feat)
			img_ids.append(img_id)
	return img_feats, img_ids

def load_feat_pair(query_folder, frontal_feats, profile_feats, frontal_ids, profile_ids):
	pair_front_feat = []
	pair_profile_feat = []
	pair_front_id = []
	pair_profile_id = []
	pair_file = 'same.txt'
	full_pair_file = os.path.join(query_folder, pair_file)
	with open(full_pair_file, 'r') as in_f:
		for line in in_f:
			record = line.strip().split(',')
			pair1, pair2 = int(record[0]), int(record[1])
			pair_front_feat.append(frontal_feats[pair1-1])
			pair_profile_feat.append(profile_feats[pair2-1])
			pair_front_id.append(frontal_ids[pair1-1])
			pair_profile_id.append(profile_ids[pair2-1])

	return pair_front_feat, pair_profile_feat, pair_front_id, pair_profile_id