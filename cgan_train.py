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
import torch.nn.functional as F

from util.util import Util

from torch.utils.data import distributed
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

optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
# optimizer_V = torch.optim.Adam(vector_net.parameters(), lr=1e-4)


adversarial_loss = torch.nn.MSELoss()
vector_net.train()
generator.train()
discriminator.train()


# Loss functions
# adversarial_loss = torch.nn.BCEWithLogitsLoss()

# Initialize generator and discriminator
# generator = Generator()
# discriminator = Discriminator()
multi_gpu = True

if cuda:
    generator.cuda()
    discriminator.cuda()
    vector_net.cuda()
    adversarial_loss.cuda()

if multi_gpu:
    generator = torch.nn.DataParallel(generator, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    discriminator = torch.nn.DataParallel(discriminator, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    vector_net = torch_geometric.nn.DataParallel(vector_net, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

    # optimizer_G = torch.nn.DataParallel(optimizer_G, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    # optimizer_D = torch.nn.DataParallel(optimizer_D, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    # optimizer_V = nn.DataParallel(optimizer_V, device_ids=[0, 1, 2, 3, 4])


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
    
    
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


for epoch in range(200):
    for i, data in enumerate(dataloader):
        if not multi_gpu:
            data = data.cuda()
            batch_size = data.num_graphs
            valid_len = data.valid_len
        else:
            batch_size = len(data)
            valid_len = torch.cat([i.valid_len for i in data]).cuda()
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).cuda()
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).cuda()

        # Configure input
        if not multi_gpu:
            real_trajs = Variable(data.agents.type(FloatTensor).view(batch_size, -1, 2)).cuda()
            labels = Variable(data.label.type(LongTensor)).cuda()
        else:
            real_trajs = torch.stack(
                [
                    i.agents for i in data
                ],
                dim=0
            ).cuda()
            labels = torch.cat([i.label.type(LongTensor) for i in data]).cuda()
            # labels = Variable(data.label.type(LongTensor)).cuda()
        h = vector_net(data)

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1000, (batch_size, 64))))
        gen_labels = Variable(LongTensor(np.random.randint(0, 8, batch_size)))

        # Generate a batch of images
        gen_trajs = generator(z, gen_labels, h.detach(), valid_len)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_trajs, gen_labels, h.detach(), valid_len)
        g_loss = adversarial_loss(validity, valid)
        loss = F.kl_div(gen_trajs.softmax(dim=-1).log(), real_trajs.softmax(dim=-1), reduction='sum') + g_loss
        loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_trajs, labels, h, valid_len)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_trajs.detach(), gen_labels, h, valid_len)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = d_real_loss*0.3 + d_fake_loss*0.7

        d_loss.backward()
        optimizer_D.step()
        # optimizer_V.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D fake loss: %f] [D real loss: %f]"
            % (epoch, 20, i, len(dataloader), d_loss.item(), g_loss.item(), d_fake_loss.item(), d_real_loss.item())
        )
    torch.save(
        generator.state_dict() if not multi_gpu else generator.module.state_dict(),
        os.path.join("tmp",f"generator_{epoch}.pt")
    )
    torch.save(
        vector_net.state_dict() if not multi_gpu else vector_net.module.state_dict(),
        os.path.join("tmp", f"vector_net_{epoch}.pt")
    )
    torch.save(
        discriminator.state_dict() if not multi_gpu else discriminator.module.state_dict(),
        os.path.join("tmp", f"discriminator_{epoch}.pt")
    )




