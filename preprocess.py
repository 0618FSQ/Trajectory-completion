from util.preprocessor.argoverse_prerocess import ArgoversePreprocessor
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import argparse


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-r", "--raw", type=str, default="../dataset")
    # parser.add_argument("-s", "--split", type=str, default="../dataset")
    # parser.add_argument("-s", "--small", action='store_true', default=False)
    # args = parser.parse_args()
    raw_dir="/data1/prediction/dataset/argoverse/csv"
    split="train"
    interm_dir="/data1/prediction/dataset/argoverse/traj_gen_preprocess"
    # file = open("result.json", "r")
    # record = file.read()
    # label_dict = json.loads(record)
    # argoverse_processor = ArgoversePreprocessor(root_dir=args.raw, split=args.split, save_dir=interm_dir)
    argoverse_processor = ArgoversePreprocessor(root_dir=raw_dir, split=split, save_dir=interm_dir)
    # argoverse_processor[0]
    loader = DataLoader(argoverse_processor,
                            batch_size=32 ,     # 1 batch in debug mode
                            num_workers=4,    # use only 0 worker in debug mode
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    for i, data in enumerate(loader):
        print(f"{i} / 3218")
