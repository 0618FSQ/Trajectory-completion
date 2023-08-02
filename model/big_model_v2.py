# VectorNet Implementation
# Author: Jianbang LIU @ RPAI Lab, CUHK
# Email: henryliu@link.cuhk.edu.hk
# Cite: https://github.com/xk-huang/yet-another-vectornet
# Modification: Add auxiliary layer and loss

import os
import random
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.global_graph import GlobalGraph, GlobalGraph2

from model.layers.basic_module import MLP
# from .vectornet import VectorNet
from torch.nn import TransformerEncoderLayer


class BigModel(nn.Module):
    
    def __init__(
        self,
        in_channels=10,
        num_subgraph_layers=3,
        num_global_graph_layer=1,
        obs_horizon=50,
        subgraph_width=64,
        global_graph_width=64
    ):
        super().__init__()
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers)
        self.out_channels = 2
        self.subgraph_width = subgraph_width
        self.global_graph_width = global_graph_width
        self.k = 1
        self.obs_horizon = obs_horizon
        self.multi_layer_mlp = nn.Sequential(
            MLP(in_channel=2, out_channel=subgraph_width, activation="leaky"),
            MLP(in_channel=subgraph_width, out_channel=subgraph_width, activation="leaky"),
            MLP(in_channel=subgraph_width, out_channel=subgraph_width, activation="leaky"),
            MLP(in_channel=subgraph_width, out_channel=subgraph_width, activation="leaky"),
            MLP(in_channel=subgraph_width, out_channel=subgraph_width, activation="leaky"),
            MLP(in_channel=subgraph_width, out_channel=subgraph_width, activation="leaky")
        )

        self.agt_encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(d_model=subgraph_width, nhead=8, batch_first=True)
                for _ in range(num_global_graph_layer)
            ]
        )

        self.global_traj = GlobalGraph2(
            in_channels=subgraph_width,
            global_graph_width=subgraph_width,
            num_global_layers=num_global_graph_layer,
            need_scale=True
        )

        self.position_encoder = nn.Parameter(
            torch.randn(self.obs_horizon, subgraph_width)
        )
        self.pred_mlp = nn.Linear(in_features=subgraph_width, out_features=2)

    def forward(self, agt_traj, global_feat, valid_len, mask):

        output = self.multi_layer_mlp(agt_traj) + self.position_encoder.unsqueeze(0)
        output = self.agt_encoder(output)
        output = self.global_traj(output, global_feat, valid_len)
        output = torch.max(output.masked_fill(~mask, -np.inf), dim=1)[0]
        output = output.unsqueeze(1) + self.position_encoder.unsqueeze(0)

        output = self.pred_mlp(output)
        return output

