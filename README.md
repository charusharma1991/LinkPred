# Learning Representations via Spectral-Biased Random Walks on Graphs

# About
This is the source code for the paper - Learning Representations via Spectral-Biased Random Walks on Graphs accepted at IJCNN 2020. We propose a novel mechanism which use random walks biased by Spectral similarity of nodes. The proposed method enhance the SOTA results on many datasets of varying attributes.

# Installation
Run the following command to install the dependencies required to run the code.
```
python -r requirements.txt
```
If your default python version is 2.X, we request you to switch to 3.X.

# Usage
The directory End2End contains the end to end pipeline to run the code. 
Run the following command to reproduce the results on Celegans dataset.
```
bash end2end.sh
```
The bash file currently has code for Celegans dataset, provided as an example. You can change the dataset name and corresponding hyperparameters from the table given in the paper to reproduce the results.
This directory has the complete pipeline, from generating the spectral walks, running the par2vec model to learn primary node embeddings and finally running the link prediction model.

The src directory contains the code just for link prediction and node classification with precomputed spectral-biased random walk and paragraph vector model output.
Run the following command to run just the link prediction part of the entire pipleline
```
python LP.py
```
The LP.py is the python script which will run the link prediction model on all datasets and store the results in a text file inside the directory result_logs/LP.

The NC.py is the python script which will run the node classification model on the datasets- cora and citeseer and store the results in a text file inside the directory result_logs/NC.
