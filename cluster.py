from collections import defaultdict
import os
import sys
import math
from tracemalloc import start
import numpy as np

from util.preprocessor.preprocessor_4_cgan import Preprocessor4CGAN
import json

import multiprocessing
from sklearn.cluster import DBSCAN, KMeans
import joblib
# features = {}

class PreprocessorWithMultiProcess(multiprocessing.Process):
    def __init__(self, path_list) :
        super().__init__()
        self.path_list = path_list
        self.r = {}
    
    def run(self):
        th_list = []
        for path in self.path_list:
            th_list.append(
                Preprocessor4CGAN(
                    path=path
                )
            )
        for th in th_list:
            th.start()
        for th in th_list:
            th.join()
        for th in th_list:
            self.r[th.file_name] = th.trans()
        print(f"多进程结果{self.r}")
        features.update(self.r)
        return 


if __name__ == "__main__":
    directory = "/home/dataset/argoverse/csv/train/"
    file_name_list = os.listdir(directory)
    length = len(file_name_list)
    
    features = {}
    for i in range(0, length, 125):
        cache = {}
        path_list = [os.path.join(directory, p) for p in file_name_list[i:i+125]]
        # length2 = len(path_list)
        th_list = []
        for path in path_list:
            th_list.append(
                Preprocessor4CGAN(
                    path=path
                )
            )
        for th in th_list:
            th.start()
        for th in th_list:
            th.join()
        for th in th_list:
            features[th.file_name] = th.trans()

    values = list(features.values())
    new_feature = np.stack(values)
    # print(new_feature)
    clustering = KMeans(8).fit(new_feature)
    result = {}
    for key, label in zip(features.keys(), clustering.labels_):
        result[key] = str(label)
    print(result)
    json_text = json.dumps(result)
    file = open("result.json", "w")
    file.write(json_text)
    joblib.dump(clustering, "cluster.bin")
