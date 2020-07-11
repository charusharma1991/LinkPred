import torch
import torch.nn.functional as F
import time
import numpy as np
from sinkhorndist import SinkhornSolver
from models import RegularizedAtt
import os

def save_model(model, epoch, dataset_name):
    print("Saving Model at epoch {}".format(epoch))

    if not os.path.exists("checkpoints"):
    	os.mkdir("checkpoints")
    if not os.path.exists("checkpoints/{}".format(dataset_name)):
    	os.mkdir("checkpoints/{}".format(dataset_name))

    torch.save(model.state_dict(),
               ("./checkpoints/{}/".format(dataset_name) + 
               	"trained_{}.pth").format(epoch))
    # print("Done saving {} Model".format(dataset_name))


spectrum_loss_func = torch.nn.MSELoss()
L1_loss_func = torch.nn.L1Loss()
ss = SinkhornSolver(epsilon=0.01, iterations=1)
reg_att = RegularizedAtt(128)
#reg_att.cuda()

def get_neighbors_average(node_embeds, neighbors, neighbors_count):
	a = node_embeds[neighbors[:, 1]].transpose(1, 0)
	edge = torch.LongTensor(np.transpose(neighbors))
	#edge = torch.LongTensor(np.transpose(neighbors)).cuda()
	b = reg_att(a, edge, node_embeds.shape[0])
	return b

def get_neighbor_spectrum_loss(iter_num, neighbors, neighbors_count, node_embeds, last_iter):
	# start = time.time()
	b = get_neighbors_average(node_embeds, neighbors, neighbors_count)
	if iter_num == last_iter - 1:
		b = b[500*iter_num:].unsqueeze(1)
		a = node_embeds[500*iter_num:].unsqueeze(1)
		#zero_tensor = torch.zeros((node_embeds.shape[0] - 500*iter_num)).cuda()
		zero_tensor = torch.zeros((node_embeds.shape[0] - 500*iter_num))
	else:
		b = b[500*iter_num : 500*(iter_num+1)].unsqueeze(1)
		a = node_embeds[500*iter_num : 500*(iter_num+1)].unsqueeze(1)
		#zero_tensor = torch.zeros((500)).cuda()
		zero_tensor = torch.zeros((500))
	
	distance, _ = ss(a, b)
	loss = L1_loss_func(distance, zero_tensor)
	# print("time taken is - ", time.time() - start)
	return loss
