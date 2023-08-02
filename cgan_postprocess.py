# from postprocess
from postprocess import PostPreprocessor
import numpy as np

import json

import pandas as pd
import os
import threading
from threading import Thread

from multiprocessing import Manager, Process,Lock

root = "dataset/train_intermediate/raw"
# csv_root = "/home/dataset/argoverse/csv/train"

trajectories = "trajectories/"
mem_dir = "cache"

        
post_processor = PostPreprocessor()

def work(file_name, key, post_processor, gen_traj, cache):
    
        origin_data = pd.read_pickle(os.path.join(root, f"features_{key}.pkl"))
        orig = origin_data['orig'].values[0]
        rot = origin_data['rot'].values[0]
        box_min = origin_data['box_min'].values[0]
        box_max = origin_data['box_max'].values[0]
        cty_name = origin_data['city'].values[0]
        real_traj = post_processor.postpreprocess(
                traj=gen_traj, 
                rot=rot, 
                orig=orig, 
                box_min=box_min, 
                box_max=box_max
            )
        sorted_real_traj = np.sort(real_traj, axis=0)
        if post_processor.cross(sorted_real_traj, cty_name):
            return
            
        sorted_real_traj = post_processor.permutate(sorted_real_traj, cty_name)
        if len(sorted_real_traj) == 0:
            return 
        cache[key] = sorted_real_traj
        print(f"{file_name}\t{key}\t{sorted_real_traj}")

if __name__ == "__main__":
    for file_name in os.listdir(trajectories):
        path = os.path.join(trajectories, file_name, f"text.json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as file:
            gen_trajs = json.loads(file.readline().strip())
            gen_labels = json.loads(file.readline().strip())
            cache = {}
            length = len(gen_trajs)
            keys = list(gen_trajs.keys())
            
            for i in range(0, length, 40):
                tmp_keys = keys[i: i+40]
                th_list = []
                
                for key in tmp_keys:
                    p=Thread(target=work, args=(file_name, key, post_processor, gen_trajs[key], cache))
                    p.start()
                    th_list.append(p)
                for p in th_list:
                    p.join(timeout=10)
                    # cache.update(tmp_cache)

            with open(os.path.join(mem_dir, f"{file_name}.json"), "w") as f:
                json_str = json.dumps(cache)
                f.write(json_str)
        