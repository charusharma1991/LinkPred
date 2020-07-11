import torch

from models import NotGAT, Par2Vec, NodeClassification
from sklearn.linear_model import LogisticRegression as LR
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from copy import deepcopy

from load_data import build_data,load_data_from_file
from create_corpus import Corpus
from utils import save_model, get_laplacian_loss, get_spectrum_loss, get_neighbor_spectrum_loss
from load_node_cl import load_node_data
import networkx as nx
from networkx import laplacian_matrix
import random
import argparse
import os
import sys
import time
import pickle


def parse_args():
	args = argparse.ArgumentParser()
	# network arguments
	args.add_argument("-data", "--data",
					  default="USAir", help="data directory")
	args.add_argument("-e", "--epochs", type=int,
					  default=4000, help="Number of epochs")
	args.add_argument("-w", "--weight_decay", type=float,
					  default=1e-8, help="L2 reglarization")
	args.add_argument("-iters", "--iterations", type=int,
					  default=1000, help="Number of iterations in training")
	args.add_argument("-b", "--batch_size", type=int,
					  default=3868, help="Batch size")
	args.add_argument("-l", "--lr", type=float, default=1e-3)
	args.add_argument("-tc", "--test_check", help="epochs after which to test it", 
					  type=int, default=3000)
	args.add_argument("-it", "--is_test", type=bool, default=False)
	args.add_argument("-split_ratio", "--split_ratio", type=float, default=90)
	# args.add_argument("-sp_loss", "--sp_loss", type=bool, default=False)

	# arguments for GAT
	args.add_argument("-drop_GAT", "--drop_GAT", type=float,
					  default=0.0, help="Dropout probability for SpGAT layer")
	args.add_argument("-alpha", "--alpha", type=float,
					  default=0.0, help="LeakyRelu alphs for SpGAT layer")
	args.add_argument("-clip", "--gradient_clip_norm", type=float,
					  default=0.25, help="maximum norm value for clipping")
	args.add_argument("-out_dim", "--node_out_dim", type=float,
					  default=[128, 64], help="Entity output embedding dimensions")
	args.add_argument("-h_gat", "--nheads_GAT", type=int,
					  default=[2,2], help="Multihead attention SpGAT")
	args.add_argument("-regterm", "--regterm",
					  default=1e-6, help="regularization term")
	args = args.parse_args()
	return args


args = parse_args()

SP_LOSS = True
L2_reg=False
L1_reg=False
def get_data(args):
	train_data, test_data, train_neg, test_neg, node_embeddings, \
		num_nodes, laplacian, spectrum, neighbors, \
		neighbors_count = build_data(args.data, args.split_ratio, args.node_out_dim[0])

	corpus = Corpus(num_nodes, train_data, test_data, train_neg, test_neg, 
					len(train_data[0]), laplacian, spectrum, neighbors, neighbors_count)
	return corpus, torch.FloatTensor(node_embeddings)

Corpus_, node_embeddings = get_data(args)

node_embeddings_copied = deepcopy(node_embeddings)

CUDA = torch.cuda.is_available()


def get_test_score(model):
	model.eval()
	with torch.no_grad():
		Corpus_.scoring(model)
		

def batch_gat_loss(gat_loss_func, train_indices, train_indices_neg, node_embeds):
	source_embeds = node_embeds[train_indices[:, 0]]
	tail_embeds = node_embeds[train_indices[:, 1]]

	value_pos = torch.sum(source_embeds * tail_embeds, dim=1)
	#target_pos = torch.ones(source_embeds.shape[0]).cuda()
	target_pos = torch.ones(source_embeds.shape[0])

	source_embeds_neg = node_embeds[train_indices_neg[:, 0]]
	tail_embeds_neg = node_embeds[train_indices_neg[:, 1]]

	value_neg = torch.sum(source_embeds_neg * tail_embeds_neg, dim=1)
	#target_neg = torch.zeros(source_embeds_neg.shape[0]).cuda()
	target_neg = torch.zeros(source_embeds_neg.shape[0])

	values = torch.cat([value_pos, value_neg], dim=-1)
	target = torch.cat([target_pos, target_neg], dim=-1)
	loss = gat_loss_func(values, target)
	# print("loss is ", loss)
	return loss

def getL():
	(train_data,train_adjacency_mat), test_data, train_neg, \
		test_neg, node_embeddings, num_nodes = build_data(args.data, args.split_ratio)
	G = np.zeros((num_nodes,num_nodes),dtype = int)
	row, col = train_adjacency_mat
	G[row, col] = 1
	G[col, row] = 1
	A = np.matrix(G)
	G = nx.from_numpy_matrix(A)
	N = laplacian_matrix(G)
	return N

def train_model(args):
	model_gat = NotGAT(node_embeddings, args.node_out_dim, args.drop_GAT,
					args.alpha, args.nheads_GAT, Corpus_.num_nodes)

	if CUDA:
		model_gat.cuda()

	if args.is_test:
		model_gat.load_state_dict(torch.load(
		  './checkpoints/{0}/trained_{1}.pth'.format(args.data, args.test_check),map_location='cpu'))
		get_test_score(model_gat)
		return
	#NN = getL()
	optimizer = torch.optim.Adam(
		model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	scheduler = torch.optim.lr_scheduler.StepLR(
		optimizer, step_size=1000, gamma=0.5, last_epoch=-1)

	# gat_loss_func = torch.nn.BCEWithLogitsLoss()
	gat_loss_func = torch.nn.MSELoss()

	epoch_losses = []   # losses of all epochs
	print("Number of epochs {}".format(args.epochs))

	model_gat.train()

	for epoch in range(args.epochs+1):
		# print("\nepoch-> ", epoch)
		# print("Training set shuffled, length is ", Corpus_.train_indices.shape)

		random.shuffle(Corpus_.train_edge_data)
		random.shuffle(Corpus_.train_neg_data)

		Corpus_.train_indices = np.array(
			list(Corpus_.train_edge_data)).astype(np.int32)
		Corpus_.train_neg_indices = np.array(
			list(Corpus_.train_neg_data)).astype(np.int32)

		start_time = time.time()
		epoch_loss = []

		if Corpus_.num_nodes % 500 == 0:
			num_iters_per_epoch = Corpus_.num_nodes // 500
		else:
			num_iters_per_epoch = (Corpus_.num_nodes // 500) + 1
		
		for iters in range(num_iters_per_epoch):
			start_time_iter = time.time()
			train_indices, train_indices_neg = Corpus_.get_iteration_batch(0)

			if CUDA:
				train_indices = Variable(torch.LongTensor(train_indices)).cuda()
				train_indices_neg = Variable(torch.LongTensor(train_indices_neg)).cuda()
			else:
				train_indices = Variable(torch.LongTensor(train_indices))

			optimizer.zero_grad()

			node_embeds = model_gat(Corpus_, Corpus_.train_adj_matrix, 
							train_indices)

			loss = batch_gat_loss(
			   gat_loss_func, train_indices, train_indices_neg, node_embeds)
			
			# laplacian_loss = get_laplacian_loss(activation_matrix, 
			# 							Corpus_.laplacian, Corpus_.num_nodes)
			# spectrum_loss = get_spectrum_loss(Corpus_.spectrum, node_embeds, 
			# 					train_indices, train_indices_neg)
			
			if SP_LOSS == True:
				neighbor_spectrum_loss = get_neighbor_spectrum_loss(iters, Corpus_.neighbors, \
											Corpus_.neighbors_count, node_embeds, num_iters_per_epoch)
				(loss + float(args.regterm)*neighbor_spectrum_loss).backward()
			if L2_reg ==  True:
				l2_reg = None
				for W in model_gat.parameters():
				    if l2_reg is None:
				        l2_reg = W.norm(2)
				    else:
			        	l2_reg = l2_reg + W.norm(2)
				(loss + float(args.regterm)*l2_reg).backward()
			if L1_reg ==  True:
				l1_reg = None
				for W in model_gat.parameters():
				    if l1_reg is None:
				        l1_reg = W.norm(1)
				    else:
			        	l1_reg = l1_reg + W.norm(1)
				(loss + float(args.regterm)*l1_reg).backward()
			else:
				loss.backward()

			optimizer.step()

			epoch_loss.append(loss.data.item())

			end_time_iter = time.time()

			# print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
			# 	iters, end_time_iter - start_time_iter, loss.data.item()))

		scheduler.step()
		# if epoch % 100 == 0:
		print("Epoch {} , average loss {} , epoch_time {}\n".format(
			epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
		epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

		if epoch>0 and epoch %100 == 0:
			save_model(model_gat, epoch, args.data)

	model_gat.load_state_dict(torch.load(
	  './checkpoints/{0}/trained_{1}.pth'.format(args.data, args.epochs)))
	get_test_score(model_gat)

train_model(args)
