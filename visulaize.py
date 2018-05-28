import os
from matplotlib import pylab as plt
import numpy as np
# from utils.convert_image import convert_image
from utils.data_loader import data_loader, data_loader_ijba, CaffeCrop, Resize
import argparse
from PIL import Image

def parse_args():
	parser = argparse.ArgumentParser(description='PF_GAN')

	parser.add_argument('-batch_size', type=int, default=32, help='batch size for training [default: 8]')
	parser.add_argument('-save_dir', type=str, default='./results', help='where to save the snapshot')
	parser.add_argument('-cuda', action='store_true', default=True, help='enable the gpu')
	# data souce
	#parser.add_argument('-random', action='store_true', default=False, help='use randomely created data to run program')
	parser.add_argument('-data_place', type=str, default='./dataset', help='prepared data path to run program')
	# model
	parser.add_argument('-model_type', type=str, default='Profile', help='train model_P')

	args = parser.parse_args()

	return args

def convert_image(data):

	img = data.transpose(0, 2, 3, 1)+1
	img = img / 2.0
	img = img * 255.
	img = img[:,:,:,[2,1,0]]

	return img.astype(np.uint8)

def main():
	args = parse_args()
	model_P = Model_P.Generator(50,3)
	path_to_model_P = 'C:\\Users\\duyson\\Desktop\\Projects\\FaceNormalize\\PytorchGAN\\snapshot\\Model_P\\2018-05-21_15-43-14\\epoch995_G.pt'
	model_P = torch.load(path_to_model_P)
	yaw_type = 'nonli'

	infos = [ ('C:\\Users\\duyson\\Desktop\\Projects\\FaceNormalize\\PytorchGAN\\dataset\\IJBA\\IJBA\\align_image_11', 'ijb_a_11_align_split', 'frame'),
			('C:\\Users\\duyson\\Desktop\\Projects\\FaceNormalize\\PytorchGAN\\dataset\\IJBA\\IJBA\\align_image_11', 'ijb_a_11_align_split', 'img'), 
			('C:\\Users\\duyson\\Desktop\\Projects\\FaceNormalize\\PytorchGAN\\dataset\\IJBA\\IJBA\\align_image_1N', 'split', 'gallery'),
			('C:\\Users\\duyson\\Desktop\\Projects\\FaceNormalize\\PytorchGAN\\dataset\\IJBA\\IJBA\\align_image_1N', 'split', 'probe') ]

	# images, id_labels, Nd, channel_num = data_loader(args.data_place, args.model_type)
	# jpg_image = convert_image(images)
	# for i in range(len(jpg_image)):
	# 	plt.imshow(jpg_image[i])
	# 	plt.show()
	if args.cuda:
		model_P.cuda()
	model_P.eval()

	for root_dir, sub_dir, img_type in infos:
		for split in range(1,11):
			split_dir = os.path.join(root_dir, sub_dir + str(split))
			img_dir = os.path.join(split_dir, img_type)
			img_list_file = os.path.join(split_dir, '{}_list_{}.txt'.format(img_type, yaw_type))

			images_ijba, channel_num = data_loader_ijba(img_dir, img_list_file)
			jpg_image = convert_image(images_ijba)
			img_dataset = IJBADataset(images_ijba,
				transforms=transforms.Compose([
				Resize((110, 110)),
				RandomCrop((96,96))
				]))

			dataloader = DataLoader(img_dataset, batch_size = args.batch_size,
				shuffle = False,
				pin_memory = True)
			Nz = 50
			for i, input_img in enumerate(dataloader):
				generated_imgs = Generate_Image_v2(input_img, model_P, Nz, args)
				if (i+1)*32 < len(jpg_image) - 1:
					show_image(jpg_image[i*32:(i+1)*32], generated_imgs, i)



if __name__ == '__main__':
	main()