import torch
import torch.nn as nn

from model.layers.basic_module import MLP
from model.backbone.vectornet import VectorNetBackbone


# from loss import VectorLoss


class VectorNet(nn.Module):

    def __init__(self,
                 in_channels=10,
                 num_subgraph_layers=3,
                 num_global_graph_layer=3,
                 subgraph_width=64,
                 global_graph_width=64
                 ):
        super(VectorNet, self).__init__()
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers)
        self.out_channels = 2
        self.subgraph_width = subgraph_width
        self.global_graph_width = global_graph_width
        self.k = 1

        self.backbone = VectorNetBackbone(
            in_channels=in_channels,
            num_subgraph_layers=num_subgraph_layers,
            subgraph_width=subgraph_width,
            num_global_graph_layer=num_global_graph_layer,
            global_graph_width=global_graph_width
        )

    def forward(self, data):
        
        global_feat = self.backbone(data)  # [batch_size, time_step_len, global_graph_width]
        
        return global_feat
