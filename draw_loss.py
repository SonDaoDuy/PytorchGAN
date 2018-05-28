import os
import numpy as np
import matplotlib.pyplot as plt

def main():
	loss_file = 'C:\\Users\\duyson\\Desktop\\Projects\\FaceNormalize\\PytorchGAN\\snapshot\\Model_P\\cfp_split_08\\Learning_Log.txt'
	D_loss, G_loss = [], []
	with open(loss_file, 'r') as in_f:
		for line in in_f:
			component = line.strip().split()
			if component[6] == 'D':
				D_loss.append(float(component[8]))
			else:
				G_loss.append(float(component[8]))

	plt.figure(1)
	t1 = np.arange(len(D_loss))
	t2 = np.arange(len(G_loss))
	plt.plot(t1, D_loss, 'r--', t2, G_loss)
	plt.show()

if __name__ == '__main__':
	main()


