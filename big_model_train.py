from model.discriminator import DiscriminatorV2
from model.generator import GeneratorV2
import torch
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from os.path import join as pjoin
from copy import deepcopy as copy
import numpy as np
from torch import nn
# from .vectornet import VectorNet
from model.vectornet import VectorNet
from model.big_model_v2 import BigModel
import torch.nn.functional as F

import torch_geometric

from dataloader.argoverse_loader import ArgoverseInMem
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
torch.cuda.set_device(0)
torch.autograd.set_detect_anomaly(True)
cuda = True
device = torch.device("cuda:0")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
BooleanTensor = torch.cuda.BoolTensor if cuda else torch.BoolTensor

# root = "/data1/prediction/dataset/argoverse/traj_gen_preprocess"
root = "/home/fsq/dataset/argoverse1"

big_model = BigModel(
    in_channels=10,
    num_subgraph_layers=3,
    num_global_graph_layer=3,
    obs_horizon=50,
    subgraph_width=256,
    global_graph_width=256
)

vectorNet = VectorNet(
    in_channels=10,
    num_subgraph_layers=3,
    num_global_graph_layer=3,
    subgraph_width=256,
    global_graph_width=256
)

print("定义模型")

# vectornet_load_path = "bigmodel/vector_net.pt"
# big_model_load_path = "bigmodel/big_model.pt"



# vectorNet.load_state_dict(
#     torch.load(vectornet_load_path, map_location=device)
# )

# big_model.load_state_dict(
#     torch.load(big_model_load_path, map_location=device)
# )

# dataset = ArgoverseInMem(pjoin(root, "train_intermediate"))
dataset = ArgoverseInMem(root)

print("加载数据完毕")
optimizer_B = torch.optim.Adam(big_model.parameters(), lr=1e-4)
optimizer_V = torch.optim.Adam(vectorNet.parameters(), lr=1e-4)


big_model.train()
vectorNet.train()

multi_gpu = False

def loss_fun(real_traj, predict_traj, mask):
    new_mask = mask.squeeze(-1)
    total_error = F.pairwise_distance(real_traj, predict_traj) 
    total_error = torch.masked_select(total_error, new_mask)
    mean_error = torch.mean(total_error)
    return mean_error

def loss_max_error(real_traj, predict_traj):
    total_error = F.pairwise_distance(real_traj, predict_traj) 
    max_error = torch.max(total_error, dim=1)[0]
    mean_error = torch.mean(max_error)
    return mean_error

def loss_kl_div(real_traj, predict_traj):
    return F.kl_div(predict_traj.softmax(dim=-1).log(), real_traj.softmax(dim=-1), reduction='sum')

def loss_error(real_traj, predict_traj):
    total_error = F.pairwise_distance(real_traj, predict_traj) 
    mean_error = torch.mean(total_error)
    return mean_error

def helper(real_traj):
    bs = real_traj.shape[0]
    mask = []
    for _ in range(bs):
        padding = np.ones(50).astype(np.bool_)
        if np.random.uniform(0, 1) < 0.9:
            start = np.random.randint(1, 49)
            if start + 1 != 49:
                end = np.random.randint(start+1, 49)
                padding[start: end + 1] = False
        mask.append(padding)
    mask = np.stack(mask)
    mask_traj = np.ma.array(
        real_traj,
        mask=np.stack([~mask] * 2, axis=-1)
    ).filled(fill_value=0)
    mask = Variable(torch.from_numpy(mask)).unsqueeze(-1)
    mask_traj = Variable(torch.from_numpy(mask_traj))
    if cuda:
        mask = mask.cuda()
        mask_traj = mask_traj.cuda()
    return mask_traj, mask
_loss = nn.MSELoss()
if cuda:
    big_model.cuda()
    vectorNet.cuda()

if multi_gpu:
    big_model = torch.nn.DataParallel(big_model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    vectorNet = torch_geometric.nn.DataParallel(vectorNet, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

    dataloader = torch_geometric.data.DataListLoader(
        dataset,
        batch_size=512, 
        shuffle=True
    )

else:
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=5,
        pin_memory=True,
        shuffle=True
    )
    
    
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


for epoch in range(200):
    for j, data in enumerate(dataloader):
        
        if not multi_gpu:
            data = data.cuda()
            batch_size = data.num_graphs
            valid_len = data.valid_len
        else:
            batch_size = len(data)
            valid_len = torch.cat([i.valid_len for i in data]).cuda()
        
        if not multi_gpu:
            real_trajs = Variable(data.agents.type(FloatTensor).view(batch_size, -1, 2), requires_grad=True).cuda()
        else:
            real_trajs = torch.stack(
                [
                    i.agents for i in data
                ],
                dim=0
            ).cuda()
        optimizer_B.zero_grad()
        optimizer_V.zero_grad()
        mask_traj, mask = helper(real_traj=real_trajs.cpu().detach().numpy())
        # if mask_traj
        
        # print(mask_traj, data)
        h = vectorNet(data)
        y = big_model(mask_traj, h, valid_len, mask)

        l1 = loss_fun(y, real_trajs, mask)
        l2 = loss_max_error(y, real_trajs)
        l3 = _loss(y, real_trajs)
        
        loss = l1 + l2 + l3
        print(f"epoch {epoch} iters {j} loss {l1.detach().cpu().item()} max loss {l2.detach().cpu().item()} mse {l3.detach().cpu().item()}")
        loss.backward()
        optimizer_B.step()
        optimizer_V.step()
    if epoch % 20 == 19:
        torch.save(
            big_model.state_dict() if not multi_gpu else big_model.module.state_dict(),
            os.path.join("bigmodel/restart",f"big_model_{epoch}.pt")
        )
        torch.save(
            vectorNet.state_dict() if not multi_gpu else vectorNet.module.state_dict(),
            os.path.join("bigmodel/restart", f"vector_net_{epoch}.pt")
        )
    