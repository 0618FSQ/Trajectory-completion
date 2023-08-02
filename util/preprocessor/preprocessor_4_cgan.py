# About: script to processing argoverse forecasting dataset
# Author: Jianbang LIU @ RPAI, CUHK
# Date: 2021.07.16

from cProfile import run
import os
import argparse
from os.path import join as pjoin
import copy
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import sparse

import warnings
import threading
import multiprocessing

warnings.filterwarnings("ignore")


class Preprocessor4CGAN(threading.Thread):
    def __init__(
        self,
        path, 
        normalized=True
    ):
        super().__init__()

        self.LANE_WIDTH = {'MIA': 3.84, 'PIT': 3.97}
        self.COLOR_DICT = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"}

        self.path = path
        self.normalized = normalized
        self.file_name = None

    def run(self):
        data = pd.read_csv(self.path)
        # seq_f_name, ext = os.path.splitext(seq_f_name_ext)
        _, self.file_name = os.path.split(self.path)
        agt_traj = self.read_argo_data(data)
        self.result = self.normalize(agt_traj)
    
    def trans(self):
        x = self.result.reshape([-1])
        return x
    

    def normalize(self, agt_traj):
        orig = np.array(agt_traj)[-1]

        # comput the rotation matrix
        if self.normalized:
            
            pre = np.array(agt_traj[-1]) - np.array(agt_traj[0])

            theta = - np.arctan2(pre[1], pre[0]) + np.pi / 2
            rot = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]], np.float32)
        else:
            # if not normalized, do not rotate.
            theta = None
            rot = np.asarray([
                [1.0, 0.0],
                [0.0, 1.0]], np.float32)

       
        agt_traj_obs = np.array(agt_traj).copy().astype(np.float32)

        # rotate the center lines and find the reference center line
        agt_traj_obs = np.matmul(rot, (agt_traj_obs - orig.reshape(-1, 2)).T).T
        return agt_traj_obs

    @staticmethod
    def read_argo_data(df: pd.DataFrame):
        # city = df["CITY_NAME"].values[0]

        """TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME"""
        agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))
        mapping = dict([(ts, i) for i, ts in enumerate(agt_ts)])

        trajs = df[["X", "Y"]].values
        steps = df['TIMESTAMP'].apply(lambda x: mapping[x]).values
        steps = [mapping[x] for x in df['TIMESTAMP'].values]
        steps = np.asarray(steps, np.int64)

        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]

        agt_idx = obj_type.index('AGENT')
        idcs = objs[keys[agt_idx]]

        agt_traj = trajs[idcs]
        return agt_traj



if __name__ == "__main__":
    path = '/home/dataset/argoverse/csv/train/63671.csv'
    preprocessor = Preprocessor4CGAN(path=path)
    preprocessor.run()