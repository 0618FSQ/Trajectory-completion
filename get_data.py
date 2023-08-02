from torch_geometric.data import DataLoader
from os.path import join as pjoin

from dataloader.argoverse_loader import ArgoverseInMem

root = "/data1/prediction/dataset/argoverse/traj_gen_preprocess"

dataset = ArgoverseInMem(pjoin(root, "train_intermediate")).shuffle()

dataloader = DataLoader(
    dataset,
    batch_size=2,
    num_workers=1,
    pin_memory=True,
    shuffle=True
    )

for iteration in dataloader:
    print(iteration)