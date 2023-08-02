import os
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.global_graph import GlobalGraph
from model.layers.subgraph import SubGraph
from model.layers.basic_module import MLP


class VectorNetBackbone(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self,
                 in_channels=8,
                 num_subgraph_layers=3,
                 num_global_graph_layer=1,
                 subgraph_width=64,
                 global_graph_width=64
                 ):
        super(VectorNetBackbone, self).__init__()
        # some params
        self.num_subgraph_layers = num_subgraph_layers
        self.global_graph_width = global_graph_width

        
        self.subgraph = SubGraph(in_channels, num_subgraph_layers, subgraph_width)

        self.global_graph = GlobalGraph(self.subgraph.out_channels,
                                        self.global_graph_width,
                                        num_global_layers=num_global_graph_layer)

    def forward(self, data):
       
        batch_size = data.num_graphs
        sub_graph_out = self.subgraph(data)
        x = sub_graph_out.view(batch_size, -1, self.subgraph.out_channels)
        valid_lens = data.valid_len

        global_graph_out = self.global_graph(x, valid_lens=valid_lens)

        return global_graph_out
