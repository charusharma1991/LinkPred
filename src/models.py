import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import SpGraphAttentionLayer, SpecialSpmmFinal 

CUDA = torch.cuda.is_available()  # checking cuda availability


class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(num_nodes, nheads * nhid,
                                             nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )

    def forward(self, Corpus_, batch_inputs, entity_embeddings, edge_list):
        x = entity_embeddings
        x = torch.cat([att(x, edge_list)
                       for att in self.attentions], dim=1)
        x = self.dropout_layer(x)
        x = F.elu(self.out_att(x, edge_list))
        return x


class NotGAT(nn.Module):
    def __init__(self, initial_entity_emb, entity_out_dim,
                 drop_GAT, alpha, nheads_GAT, num_nodes):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        
        self.drop_GAT = drop_GAT
        self.alpha = alpha      # For leaky relu
        
        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1))

        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        
        self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1)

        # self.spectrum_layer = nn.Linear(30, self.entity_out_dim_1)
        self.W_entities = nn.Parameter(torch.zeros(size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)
        self.sigmoid = nn.Sigmoid()

    def forward(self, Corpus_, adj, batch_inputs):
        # getting edge list
        # edge_list = adj

        # if(CUDA):
        #     edge_list = edge_list.cuda()

        # start = time.time()

        # out_entity_1 = self.sparse_gat_1(Corpus_, 
        # 				batch_inputs, self.entity_embeddings, edge_list)

        # mask_indices = torch.unique(batch_inputs[:, 1]).cuda()
        # mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()
        # mask[mask_indices] = 1.0

        # entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        # out_entity_1 = entities_upgraded + \
        #     mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        # spectrum_out = self.spectrum_layer(spectrum)
        # output = torch.cat([spectrum_out, self.entity_embeddings], dim=1)
        out_entity_1 = F.normalize(self.entity_embeddings, p=2, dim=1)
        self.final_entity_embeddings.data = out_entity_1.data
        return out_entity_1


class Par2Vec(nn.Module):
    def __init__(self, walk_embeddings, out_dim, drop_conv, out_channels,
                    walks_per_node, in_channels, num_nodes):
        super().__init__()

        walk_embeddings = walk_embeddings.transpose(1, 2)
        walk_embeddings = walk_embeddings.unsqueeze(1)
        self.batch_size = walk_embeddings.shape[0]

        self.walk_embeddings = nn.Parameter(walk_embeddings, requires_grad=True)
        self.node_embeddings = nn.Parameter(
            torch.randn(num_nodes, out_dim))

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, walks_per_node)) 

        self.dropout = nn.Dropout(drop_conv)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((out_dim) * out_channels, out_dim)

        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)

    def forward(self):

        out_conv = self.dropout(self.conv_layer(self.walk_embeddings))
        output = self.fc_layer(out_conv.squeeze(-1).view(self.batch_size, -1))
        output = F.normalize(output, p=2, dim=1)
        self.node_embeddings.data = output
        return output


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
        # powers = -self.leakyrelu(self.W.mm(inputs).squeeze())
        # edge_e = torch.exp(powers).unsqueeze(1)
        # # print(edge_e)
        # assert not torch.isnan(edge_e).any()
        # # edge_e: E

        # # e_rowsum = self.special_spmm_final(
        # #     edge, edge_e, N, edge_e.shape[0], 1)
        # # e_rowsum[e_rowsum == 0.0] = 1e-12

        # # e_rowsum: N x 1
        # edge_e = edge_e.squeeze(1)
        # edge_e = self.dropout(edge_e)
        # # edge_e: E
        # edge_w = (edge_e * inputs).t()
        # edge_w: E * D

        edge_w = inputs.transpose(1, 0)
        h_prime = self.special_spmm_final(
            edge, edge_w, N, edge_w.shape[0], self.dimensions)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        # h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        return h_prime


class NodeClassification(nn.Module):
    def __init__(self, node_embeddings, num_classes, alpha, drop_prob):
        super().__init__()
        self.node_embeddings = nn.Parameter(node_embeddings)
        self.num_nodes = node_embeddings.shape[0]
        self.dimensions = node_embeddings.shape[1]

        self.fc1 = nn.Linear(self.dimensions, self.dimensions//2)
        self.fc2 = nn.Linear(self.dimensions//2, num_classes)
        self.non_linearity = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(drop_prob)

    def forward(self, train_indices):
        return self.fc2(self.dropout_layer(self.fc1(self.node_embeddings[train_indices])))