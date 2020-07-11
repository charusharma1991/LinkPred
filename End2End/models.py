import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import SpecialSpmmFinal

CUDA = torch.cuda.is_available()  # checking cuda availability

class Model(nn.Module):
    def __init__(self, initial_entity_emb, entity_out_dim):

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_out_dim_1 = entity_out_dim[0]
        
        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1))

        self.entity_embeddings = nn.Parameter(initial_entity_emb)

    def forward(self):
        out_entity_1 = F.normalize(self.entity_embeddings, p=2, dim=1)
        self.final_entity_embeddings.data = out_entity_1.data
        return out_entity_1

class RegularizedAtt(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.dimensions = dimensions
        self.dropout = nn.Dropout(0)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.W = nn.Parameter(torch.zeros(size=(1, dimensions)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, inputs, edge, N):
        edge_w = inputs.transpose(1, 0)
        h_prime = self.special_spmm_final(
            edge, edge_w, N, edge_w.shape[0], self.dimensions)
        assert not torch.isnan(h_prime).any()
        return h_prime