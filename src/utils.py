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
	#edge = torch.LongTensor(np.transpose(neighbors)).cuda()
	edge = torch.LongTensor(np.transpose(neighbors))
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


def get_spectrum_loss(spectrum, node_embeds, train_indices, train_indices_neg):
	spectrum = F.normalize(spectrum, p=2, dim=1)

	source_embeds = node_embeds[train_indices[:, 0]]
	tail_embeds = node_embeds[train_indices[:, 1]]
	value = torch.sum(source_embeds * tail_embeds, dim=1)
	
	source_embeds_spectrum = spectrum[train_indices[:, 0]]
	tail_embeds_spectrum = spectrum[train_indices[:, 1]]
	target = torch.sum(source_embeds_spectrum * tail_embeds_spectrum, dim=1)

	loss = spectrum_loss_func(value, target)
	return loss


def get_laplacian_loss(activation_matrix, laplacian, num_nodes):
	#zero_tensor = torch.zeros((num_nodes)).cuda()
	zero_tensor = torch.zeros((num_nodes))
	out1 = torch.matmul(laplacian, activation_matrix.transpose(1, 0))
	out1 = out1.transpose(1, 0)
	out2 = torch.sum(activation_matrix * out1, dim=1)
	loss = loss_func(out2, zero_tensor)
	return loss
