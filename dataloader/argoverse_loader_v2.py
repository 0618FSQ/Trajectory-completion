import sys
import os
import os.path as osp
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import gc
from copy import deepcopy, copy

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
# from torch.utils.data import DataLoader

# sys.path.append("core/dataloader")


def get_fc_edge_index(node_indices):
    """
    node_indices: np.array([indices]), the indices of nodes connecting with each other;
    return a tensor(2, edges), indicing edge_index
    """
    xx, yy = np.meshgrid(node_indices, node_indices)
    xy = np.vstack(([xx.reshape(-1), yy.reshape(-1)])).astype(np.int64)
    return xy


def get_traj_edge_index(node_indices):
    """
    generate the polyline graph for traj, each node are only directionally connected with the nodes in its future
    node_indices: np.array([indices]), the indices of nodes connecting with each other;
    return a tensor(2, edges), indicing edge_index
    """
    edge_index = np.empty((2, 0))
    for i in range(len(node_indices)):
        xx, yy = np.meshgrid(node_indices[i], node_indices[i:])
        edge_index = np.hstack([edge_index, np.vstack(([xx.reshape(-1), yy.reshape(-1)])).astype(np.int64)])
    return edge_index


class GraphData(Data):
    """
    override key `cluster` indicating which polyline_id is for the vector
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.x.size(0)
        elif key == 'cluster':
            return int(self.cluster.max().item()) + 1
        else:
            return 0


# dataset loader which loads data into memory
class ArgoverseInMem(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ArgoverseInMem, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        gc.collect()

    @property
    def raw_file_names(self):
        return [file for file in os.listdir(self.raw_dir) if "features" in file and file.endswith(".pkl")]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        """ transform the raw data and store in GraphData """
        # loading the raw data
        file = open("text.txt", "w")
        traj_lens = []
        valid_lens = []
        candidate_lens = []
        for raw_path in tqdm(self.raw_paths, desc="Loading Raw Data..."):
            raw_data = pd.read_pickle(raw_path)

            # statistics
            traj_num = raw_data['feats'].values[0].shape[0]
            traj_lens.append(traj_num)

            lane_num = raw_data['graph'].values[0]['lane_idcs'].max() + 1
            valid_lens.append(traj_num + lane_num)

            candidate_num = raw_data['tar_candts'].values[0].shape[0]
            candidate_lens.append(candidate_num)
        num_valid_len_max = np.max(valid_lens)
        num_candidate_max = np.max(candidate_lens)
        print("\n[Argoverse]: The maximum of val_intermediate length is {}.".format(num_valid_len_max))
        print("[Argoverse]: The maximum of no. of candidates is {}.".format(num_candidate_max))

        # pad vectors to the largest polyline id and extend cluster, save the Data to disk
        data_list = []
        for ind, raw_path in enumerate(tqdm(self.raw_paths, desc="Transforming the data to GraphData...")):
            raw_data = pd.read_pickle(raw_path)

            # input data
            # try:
            # feats, agt_obs, label, cluster, edge_index, identifier
            try:
                x, agents, label, cluster, edge_index, identifier = self._get_x(raw_data)
                    # y = self._get_y(raw_data)
                graph_input = GraphData(
                        x=torch.from_numpy(x).float(),
                        label = torch.Tensor([label]).int(),
                        agents=torch.from_numpy(agents).float(),
                        cluster=torch.from_numpy(cluster).short(),
                        edge_index=torch.from_numpy(edge_index).long(),
                        identifier=torch.from_numpy(identifier).float(),    # the identify embedding of global graph completion

                        traj_len=torch.tensor([traj_lens[ind]]).int(),            # number of traj polyline
                        valid_len=torch.tensor([valid_lens[ind]]).int(),          # number of val_intermediate polyline
                        time_step_len=torch.tensor([num_valid_len_max]).int(),    # the maximum of no. of polyline

                        candidate_len_max=torch.tensor([num_candidate_max]).int(),
                        candidate=torch.from_numpy(raw_data['tar_candts'].values[0]).float(),
                        box_min = torch.from_numpy(raw_data['box_min'].values[0]).float(),
                        box_max = torch.from_numpy(raw_data['box_max'].values[0]).float(),
                        orig=torch.from_numpy(raw_data['orig'].values[0]).float().unsqueeze(0),
                        rot=torch.from_numpy(raw_data['rot'].values[0]).float().unsqueeze(0),
                        seq_id=torch.tensor([int(raw_data['seq_id'])]).int(),


                    )
                data_list.append(graph_input)
            except:
                file.write(f"{raw_path} \n")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        data = super(ArgoverseInMem, self).get(idx).clone()

        feature_len = data.x.shape[1]
        index_to_pad = data.time_step_len[0].item()
        valid_len = data.valid_len[0].item()
        
        data.x = torch.cat([data.x, torch.zeros((index_to_pad - valid_len, feature_len), dtype=data.x.dtype)])
        data.cluster = torch.cat([data.cluster, torch.arange(valid_len, index_to_pad, dtype=data.cluster.dtype)]).long()
        data.identifier = torch.cat([data.identifier, torch.zeros((index_to_pad - valid_len, 2), dtype=data.identifier.dtype)])


        # pad candidate and candidate_gt
        num_cand_max = data.candidate_len_max[0].item()
        data.candidate_mask = torch.cat([torch.ones((len(data.candidate), 1)),
                                         torch.zeros((num_cand_max - len(data.candidate), 1))])
        data.candidate = torch.cat([data.candidate[:, :2], torch.zeros((num_cand_max - len(data.candidate), 2))])

        seq_id = data.seq_id.item()
        path = os.path.join(self.root, "raw", f"features_{seq_id}.pkl")
        raw_data = pd.read_pickle(path)
        data.label = torch.Tensor([raw_data["label"].values[0]]).int()
        assert data.cluster.shape[0] == data.x.shape[0], "[ERROR]: Loader error!"

        return data

    @staticmethod
    def _get_x(data_seq):
        
        feats = np.empty((0, 10))
        edge_index = np.empty((2, 0), dtype=np.int64)
        identifier = np.empty((0, 2))
        print(data_seq["seq_id"].values[0])
        # get traj features
        traj_feats = data_seq['feats'].values[0]
        step = np.arange(0, traj_feats.shape[1]).reshape((-1, 1))
        traj_cnt = 0
        for _, feat in enumerate(traj_feats):
            xy_s = feat[:-1, :2]
            vec = feat[1:, :2] - feat[:-1, :2]
            traffic_ctrl = np.zeros((len(xy_s), 1))
            is_intersect = np.zeros((len(xy_s), 1))
            is_turn = np.zeros((len(xy_s), 2))
            polyline_id = np.ones((len(xy_s), 1)) * traj_cnt
            feats = np.vstack([feats, np.hstack([xy_s, vec, step[:-1], traffic_ctrl, is_turn, is_intersect, polyline_id])])
            traj_cnt += 1

        # get lane features
        graph = data_seq['graph'].values[0]
        ctrs = graph['ctrs']
        vec = graph['feats']
        traffic_ctrl = graph['control'].reshape(-1, 1)
        is_turns = graph['turn']
        is_intersect = graph['intersect'].reshape(-1, 1)
        lane_idcs = graph['lane_idcs'].reshape(-1, 1) + traj_cnt
        steps = np.zeros((len(lane_idcs), 1))
        feats = np.vstack([feats, np.hstack([ctrs, vec, steps, traffic_ctrl, is_turns, is_intersect, lane_idcs])])
        agt_obs = data_seq['agt_traj'].values[0]
        label = data_seq["label"].values[0]
        # get the cluster and construct subgraph edge_index
        cluster = copy(feats[:, -1].astype(np.int64))
        for cluster_idc in np.unique(cluster):
            [indices] = np.where(cluster == cluster_idc)
            identifier = np.vstack([identifier, np.min(feats[indices, :2], axis=0)])
            if len(indices) <= 1:
                continue                # skip if only 1 node
            if cluster_idc < traj_cnt:
                edge_index = np.hstack([edge_index, get_traj_edge_index(indices)])
            else:
                edge_index = np.hstack([edge_index, get_fc_edge_index(indices)])
        return feats, agt_obs, label, cluster, edge_index, identifier
