import torch
import numpy as np
from collections import defaultdict
import time
import queue
import random
from sklearn import metrics


class Corpus:
	def __init__(self, num_nodes, train_data, test_data, train_neg_data, 
					test_neg_data, batch_size, laplacian, spectrum, neighbors, neighbors_count):
		self.num_nodes = num_nodes
		self.train_edge_data = train_data[0]

		self.train_adj_matrix = torch.LongTensor(
			[train_data[1][0], train_data[1][1]])  # rows and columns
		
		# print("number of nodes is - ", num_nodes)
		# print("Number of edges is - ", len(train_data[0]))
		self.test_edge_data = test_data[0]
		
		# self.laplacian = torch.FloatTensor(laplacian).cuda()
		# self.spectrum = torch.FloatTensor(spectrum).cuda()
		self.neighbors = neighbors
		#self.neighbors_count = torch.FloatTensor(neighbors_count).cuda()
		self.neighbors_count = torch.FloatTensor(neighbors_count)
		self.batch_size = batch_size

		# positive training and test indices
		self.train_indices = np.array(
			list(self.train_edge_data)).astype(np.int32)

		self.test_indices = np.array(list(self.test_edge_data)).astype(np.int32)

		self.train_neg_data = train_neg_data
		self.test_neg_data = test_neg_data
		# negative training and test indices
		self.train_neg_indices = np.array(
			list(self.train_neg_data)).astype(np.int32)

		self.test_neg_indices = np.array(list(self.test_neg_data)).astype(np.int32)

	def get_iteration_batch(self, iter_num):
		if (iter_num + 1) * self.batch_size <= len(self.train_indices):
			self.batch_indices = np.empty(
				(self.batch_size, 2)).astype(np.int32)

			indices = range(self.batch_size * iter_num, self.batch_size * (iter_num + 1))

			self.batch_indices[:self.batch_size,
							   :] = self.train_indices[indices, :]

			self.batch_indices_neg = np.empty(
				(self.batch_size, 2)).astype(np.int32)

			indices_neg = range(self.batch_size * iter_num, self.batch_size * (iter_num + 1))

			self.batch_indices_neg[:self.batch_size,
							   :] = self.train_neg_indices[indices_neg, :]

			return self.batch_indices, self.batch_indices_neg

		else:
			last_iter_size = len(self.train_indices) - self.batch_size * iter_num
			self.batch_indices = np.empty(
				(last_iter_size, 2)).astype(np.int32)

			indices = range(self.batch_size * iter_num,
							len(self.train_indices))

			self.batch_indices[:last_iter_size,
							   :] = self.train_indices[indices, :]

			self.batch_indices_neg = np.empty(
				(last_iter_size, 2)).astype(np.int32)

			indices_neg = range(self.batch_size * iter_num,
							len(self.train_neg_indices))

			self.batch_indices_neg[:last_iter_size,
							   :] = self.train_neg_indices[indices_neg, :]

			return self.batch_indices, self.batch_indices_neg

	def scoring(self, model):
		source_pos = model.final_entity_embeddings[self.test_indices[:, 0]]
		tail_pos = model.final_entity_embeddings[self.test_indices[:, 1]]
		score_pos = torch.sum(source_pos * tail_pos, dim=1)
		
		source_neg = model.final_entity_embeddings[self.test_neg_indices[:, 0]]
		tail_neg = model.final_entity_embeddings[self.test_neg_indices[:, 1]]
		score_neg = torch.sum(source_neg * tail_neg, dim=1)
		
		scores = np.concatenate([score_pos.cpu().numpy(), score_neg.cpu().numpy()])
		# print(scores)
		labels = np.hstack([np.ones(score_pos.shape[0]), np.zeros(score_neg.shape[0])])
		self.AUCscore(labels, scores)

	def AUCscore(self, labels, scores):
		fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
		auc = metrics.auc(fpr, tpr)
		print("The AUC value is - ", auc)
