from model.discriminator import DiscriminatorV2
from model.generator import GeneratorV2
import torch
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from os.path import join as pjoin
from copy import deepcopy as copy
import numpy as np
from torch import nn
import json
# from .vectornet import VectorNet
from model.vectornet import VectorNet
import torch.nn.functional as F

from util.util import Util

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

root = "dataset/"

dataset = ArgoverseInMem(pjoin(root, "train_intermediate")).shuffle()

generator = GeneratorV2(
    input_size=64, hidden_size=256, n_classes=8, steps=50
)
vector_net = VectorNet()
discriminator = DiscriminatorV2(
    
        subgraph_width=64,
        
        n_classes=8
)

vectornet_load_path = "tmp/vector_net_199.pt"
generator_load_path = "tmp/generator_199.pt"
discriminator_load_path = "tmp/discriminator_199.pt"
vector_net.load_state_dict(
    torch.load(vectornet_load_path, map_location=device)
)

generator.load_state_dict(
    torch.load(generator_load_path, map_location=device)
)

discriminator.load_state_dict(
    torch.load(discriminator_load_path, map_location=device)
)

multi_gpu = True

if cuda:
    generator.cuda()
    discriminator.cuda()
    vector_net.cuda()

if multi_gpu:
    generator = nn.DataParallel(generator, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    discriminator = nn.DataParallel(discriminator, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    vector_net = torch_geometric.nn.DataParallel(vector_net, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

    dataloader = torch_geometric.data.DataListLoader(
        dataset,
        batch_size=1024, 
        shuffle=True
    )

else:
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=5,
        pin_memory=True,
        shuffle=True
    )
    
generator.eval()
discriminator.eval()
vector_net.eval()
    
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor



# ----------
#  Training
# ----------
directory = "trajectories"
with torch.no_grad():
    for epoch in range(200):
        if not os.path.exists(os.path.join(directory, f"{epoch}")):
            os.mkdir(os.path.join(directory, f"{epoch}"))
        cache = {}
        traj_labels = {}
        for i, data in enumerate(dataloader):
            if not multi_gpu:
                data = data.cuda()
                batch_size = data.num_graphs
                valid_len = data.valid_len
            else:
                batch_size = len(data)
                valid_len = torch.cat([i.valid_len for i in data]).cuda()
            
            if not multi_gpu:
                real_trajs = Variable(data.agents.type(FloatTensor).view(batch_size, -1, 2)).cuda()
                labels = Variable(data.label.type(LongTensor)).cuda()
            else:
                real_trajs = torch.stack([i.agents for i in data],dim=0).cuda()
                labels = torch.cat([i.label.type(LongTensor) for i in data]).cuda()
            h = vector_net(data)

            z = Variable(FloatTensor(np.random.normal(0, 1000, (batch_size, 64))))
            gen_labels = Variable(LongTensor(np.random.randint(0, 8, batch_size)))
            gen_trajs = generator(z, gen_labels, h.detach(), valid_len)
            seq_id = [item.seq_id.cpu().numpy().tolist()[0] for item in data]
            new_gen_traj = gen_trajs.cpu().detach().numpy().tolist()
            new_labels = gen_labels.cpu().detach().numpy().tolist()
            cache.update(dict(zip(seq_id, new_gen_traj)))
            traj_labels.update(dict(zip(seq_id, new_labels)))
        with open(os.path.join(directory, f"{epoch}", "text.json"), "w") as file:
            json_str = json.dumps(cache)
            file.write(json_str+ "\n")
            json_str2 = json.dumps(traj_labels)
            file.write(json_str2)


            