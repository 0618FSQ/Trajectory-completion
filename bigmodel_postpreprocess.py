# from postprocess
from postprocess import PostPreprocessor
import numpy as np

import json

import pandas as pd
import os
import threading
from threading import Thread

from multiprocessing import Manager, Process,Lock

root = "/data1/prediction/dataset/argoverse/traj_gen_preprocess/train_intermediate/raw/"
# csv_root = "/home/dataset/argoverse/csv/train"

trajectories = "trajectories2/"

        
post_processor = PostPreprocessor()

def work(key, gen_traj):
    origin_data = pd.read_pickle(os.path.join(root, f"features_{key}.pkl"))
    orig = origin_data['orig'].values[0]
    rot = origin_data['rot'].values[0]
    box_min = origin_data['box_min'].values[0]
    box_max = origin_data['box_max'].values[0]
    real_traj = post_processor.postpreprocess(
            traj=gen_traj, 
            rot=rot, 
            orig=orig, 
            box_min=box_min, 
            box_max=box_max
        )
    
    real = post_processor.postpreprocess(
            traj=origin_data['agt_traj'][0], 
            rot=rot, 
            orig=orig, 
            box_min=box_min, 
            box_max=box_max
        )
    print(real - real_traj)
    return real_traj

if __name__ == "__main__":
    
        path = os.path.join(trajectories, f"text_v7.json")
        
        with open(path, "r") as file:
            gen_trajs = json.loads(file.readline().strip())
            
            cache = {}
            for key, traj in gen_trajs.items():

                real_traj = work(key, traj)
                cache[key] = real_traj.tolist()

            with open(f"test.json", "w")as f:
                json_str = json.dumps(cache)
                f.write(json_str)
        