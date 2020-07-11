import numpy as np
from scipy.io import loadmat

def create_train_matrix(train_edge_data, num_nodes):
	x = np.zeros((num_nodes, num_nodes)).astype(np.float32)
	for i in range(len(train_edge_data)):
		x[train_edge_data[i][0], train_edge_data[i][1]] = 1.0
	return x

def load_neighbors(name, num_nodes, train_edge_data):
	# x = loadmat('../data/mat_versions/{}.mat'.format(name))
	# b = x['net']
	# b = b.toarray()
	b = create_train_matrix(train_edge_data, num_nodes)

	links = np.transpose(np.nonzero(b))
	# print(np.sum(b, axis=1))
	return links, np.sum(b, axis=1)
	# print(links.shape)
	neighbors = {}
	j = 0
	for i in range(num_nodes):
		if i in neighbors.keys():
			k = j
			if k>=links.shape[0]:
				break
			while(links[k, 0] == i):
				neighbors[i].append(links[k, 1])
				k += 1
				if(k >= links.shape[0]):
					break
			j = k

		else:
			neighbors[i] = []
			k = j
			if k>=links.shape[0]:
				break
			while(links[k, 0] == i):
				neighbors[i].append(links[k, 1])
				k += 1
				if(k >= links.shape[0]):
					break
			j = k
		if neighbors[i]==[]:
			neighbors[i]=[i]
	return neighbors

# neighbors = load_neighbors("Celegans", 297)
# print(neighbors)
