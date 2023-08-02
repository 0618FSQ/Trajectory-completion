from turtle import forward
import torch.nn as nn

from .vectornet import VectorNet
from model.layers.basic_module import MLP
from model.layers.global_graph import SelfAttentionWithKeyFCLayer
import torch

class Discriminator(nn.Module):
    def __init__(
        self, 
        subgraph_width,
        n_classes
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, subgraph_width)
        self.traj_encode = MLP(
            in_channel=2, out_channel=subgraph_width
        )

        # self.self_attn_key_fc = SelfAttentionWithKeyFCLayer(
        #     in_channels=subgraph_width, 
        #     global_graph_width=subgraph_width, 
        #     need_scale=True
        # )
        self.module = nn.Sequential(
            nn.Linear(subgraph_width, 1),
            nn.Sigmoid()
        )
        
        self.mlp = MLP(
            in_channel=2, out_channel=subgraph_width, activation="leaky"
        )
    
    def forward(self, traj, labels):
        x = self.mlp(traj)
        label_embedding = self.embedding(labels)
        # x = x + label_embedding.unsqueeze(1)
        # traj_encode = self.self_attn_key_fc(x, h, valid_len)
        traj_max_pool = torch.max(x, dim=1)[0] + label_embedding

        return self.module(traj_max_pool)


class DiscriminatorV2(nn.Module):
    def __init__(
        self, 
        subgraph_width,
        n_classes
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, subgraph_width)
        self.traj_encode = MLP(
            in_channel=2, out_channel=subgraph_width
        )

        self.self_attn_key_fc = SelfAttentionWithKeyFCLayer(
            in_channels=subgraph_width, 
            global_graph_width=subgraph_width, 
            need_scale=True
        )
        self.module = nn.Sequential(
            nn.Linear(subgraph_width, 1),
            nn.Sigmoid()
        )
        
        self.mlp = MLP(
            in_channel=2, out_channel=subgraph_width, activation="leaky"
        )
    
    def forward(self, traj, labels, h, valid_len):
        x = self.mlp(traj)
        label_embedding = self.embedding(labels)
        x = x + label_embedding.unsqueeze(1)
        traj_encode = self.self_attn_key_fc(x, h, valid_len)
        traj_max_pool = torch.max(traj_encode, dim=1)[0] + label_embedding

        return self.module(traj_max_pool)

# class Discriminator(nn.Module):
#     def __init__(
#         self, 
#         in_channels,
#         num_subgraph_layers,
#         num_global_graph_layer,
#         subgraph_width,
#         global_graph_width,
#         n_classes
        
#     ):
#         super().__init__()
#         self.vector_net = VectorNet(
#             in_channels,
#             num_subgraph_layers,
#             num_global_graph_layer,
#             subgraph_width,
#             global_graph_width
#         )
#         self.embedding = nn.Embedding(n_classes, subgraph_width)
#         self.traj_encode = MLP(
#             in_channel=2, out_channel=subgraph_width
#         )

#         self.self_attn_key_fc = SelfAttentionWithKeyFCLayer(
#             in_channels=subgraph_width, 
#             global_graph_width=subgraph_width, 
#             need_scale=True
#         )
#         self.module = nn.Sequential(
#             nn.Linear(subgraph_width, 1),
#             nn.Sigmoid()
#         )
        
#         self.mlp = MLP(
#             in_channel=2, out_channel=subgraph_width, activation="leaky"
#         )

    
#     def forward(self, gen_traj, lables, data, is_generate=False):
#         output = self.vector_net(data)
#         if not is_generate:
#             agents = data.agents
#             bs = data.num_graphs
#             new_agents = agents.view(bs, -1, 2)
#             x = self.mlp(new_agents)
            
#             label_embedding = self.embedding(data.label)
#         else:
#             agents = gen_traj
#             x = self.mlp(agents)
#             label_embedding = self.embedding(lables)
#         # x = x + label_embedding.unsqueeze(1)
#         traj_encode = self.self_attn_key_fc(x, output, data.valid_len)
#         traj_max_pool = torch.max(traj_encode, dim=1)[0] + label_embedding

#         return self.module(traj_max_pool)
