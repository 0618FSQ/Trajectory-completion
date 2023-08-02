import torch
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from os.path import join as pjoin
import numpy as np
from torch import nn
import json
import torch
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

root = "/data1/prediction/dataset/argoverse/traj_gen_preprocess"

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


# big_model_params = sum([param.nelement() for param in big_model.parameters()])
# vectornet_params = sum([param.nelement() for param in vectorNet.parameters()])
# print("Number of parameter: %.2fM" % ((big_model_params + vectornet_params)/1e6))


vectornet_load_path = "bigmodel/restart/vector_net_79.pt"
big_model_load_path = "bigmodel/restart/big_model_79.pt"

vectorNet.load_state_dict(
    torch.load(vectornet_load_path, map_location='cpu')
)

big_model.load_state_dict(
    torch.load(big_model_load_path, map_location='cpu')
)

dataset = ArgoverseInMem(pjoin(root, "valid_intermediate")).shuffle()

multi_gpu = False

if cuda:
    vectorNet.cuda()
    big_model.cuda()

if multi_gpu:
    big_model = nn.DataParallel(big_model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    vectorNet = torch_geometric.nn.DataParallel(vectorNet, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

    dataloader = torch_geometric.data.DataListLoader(
        dataset,
        batch_size=1024,
        shuffle=True
    )

else:
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=5,
        pin_memory=True,
        shuffle=True
    )

big_model.eval()
vectorNet.eval()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def helper(real_traj):
    bs = real_traj.shape[0]
    mask = []
    for _ in range(bs):
        padding = np.ones(50).astype(np.bool_)
        padding[1:-1] = False
        # padding[20:] = False

        # if np.random.uniform(0, 1) < 0.9:
        #     start = np.random.randint(1, 49)
        #     if start + 1 != 49:
        #         end = np.random.randint(start + 1, 49)
        #         padding[start: end + 1] = False
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


def process(real_traj, prediction_traj, mask):
    traj = real_traj.masked_fill(~mask, 0) + prediction_traj.masked_fill(mask, 0)
    return traj


directory = "trajectories2"
cache = {}
with torch.no_grad():
    for j, data in enumerate(dataloader):

        if not multi_gpu:
            data = data.cuda()
            batch_size = data.num_graphs
            valid_len = data.valid_len
            seq_id = data.seq_id
        else:
            batch_size = len(data)
            valid_len = torch.cat([i.valid_len for i in data]).cuda()
            seq_id = torch.cat([i.seq_id for i in data]).cuda()

        if not multi_gpu:
            real_trajs = Variable(data.agents.type(FloatTensor).view(batch_size, -1, 2), requires_grad=True).cuda()
        else:
            real_trajs = torch.stack(
                [
                    i.agents for i in data
                ],
                dim=0
            ).cuda()

        mask_traj, mask = helper(real_traj=real_trajs.cpu().detach().numpy())
        
        h = vectorNet(data)
        print(h.shape)
        y = big_model(mask_traj, h, valid_len, mask)
        # traj = process(real_traj=real_trajs, prediction_traj=y, mask=mask)
        print(j,y.shape[0])
        cache.update(dict(zip(seq_id.cpu().numpy().tolist(), y.cpu().numpy().tolist())))
        
    with open(os.path.join(directory, "restart.json"), "w") as file:
        json_str = json.dumps(cache)
        file.write(json_str+ "\n")
