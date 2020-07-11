from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import sys

def load_walks(dataset_name):
	all_walks = []
	with open('walks/{0}_SW_50_4.txt'.format(dataset_name), 'r') as f:
		for line in f:
			line = line.strip().split()
			current_walk = [ str(word.strip()) for word in line ]
			all_walks.append(current_walk)
	# print("number of walks is - ", len(all_walks))
	return all_walks

dataset_name = sys.argv[1]

all_walks = load_walks(dataset_name)
# print("Current dataset is - ", dataset_name)

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_walks)]

import time
start_time = time.time()

model = Doc2Vec(documents, vector_size=128, window=10, min_count=0, 
				 hs=1, epochs=50, workers=20)
model.save("par2vec/{0}_SW_emb_50_4.model".format(dataset_name))
# print("time taken - ", time.time() - start_time)
