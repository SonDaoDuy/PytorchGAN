#!/usr/bin/env python
# encoding: utf-8

def log_result(text, save_dir):
	with open('{}/Recognition_Result.txt'.format(save_dir),'a') as f:
		f.write("{}\n".format(text))
