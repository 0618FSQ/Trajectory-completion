from matplotlib import pyplot as plt
from scipy import sparse
from collections import defaultdict
import pandas as pd
import numpy as np
import imageio
import os
import json
from argoverse.utils.mpl_plotting_utils import visualize_centerline



class View:
    def vis_background(self, graph):
        lines_ctrs = graph['ctrs']
        lines_feats = graph['feats']
        lane_idcs = graph['lane_idcs']
        for i in np.unique(lane_idcs):
            line_ctr = lines_ctrs[lane_idcs == i]
            line_feat = lines_feats[lane_idcs == i]
            line_str = (2.0 * line_ctr - line_feat) / 2.0
            line_end = (2.0 * line_ctr[-1, :] + line_feat[-1, :]) / 2.0
            line = np.vstack([line_str, line_end.reshape(-1, 2)])
            # visualize_centerline(line)
            line_coords = list(zip(*line))
            lineX = line_coords[0]
            lineY = line_coords[1]
            plt.plot(lineX, lineY, "--", color="grey", alpha=1, linewidth=1, zorder=0)
            # plt.text(lineX[0], lineY[0], "s")
            # plt.text(lineX[-1], lineY[-1], "e")
            plt.axis("equal")
        # plt.savefig('./111.jpg')

    def plot_sct_traj(self, obs, traj_id=None):
        # assert len(obs) != 0, "ERROR: The input trajectory is empty!"
        traj_na = "t{}".format(traj_id) if traj_id else "traj"
        obj_type = "AGENT" if traj_id == 0 else "OTHERS"
        COLOR_DICT_2 = {"AGENT":"#FFFF00", "OTHERS":"#00FFFF"}

        plt.scatter(obs[0], obs[1], color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)
        
    def plot_frame_track(self, data, frame):
        trajs = data['feats'][:, :, :2]
        has_obss = data['has_obss']
        preds = data['gt_preds']
        has_preds = data['has_preds']
        if frame < self.obs_horizon:
            for i, [traj, has_obs] in enumerate(zip(trajs, has_obss)):
                if has_obs[frame]:
                    self.plot_sct_traj(traj[frame], i)
        if self.obs_horizon <= frame < self.obs_horizon + self.pred_horizon:
            frame -= self.obs_horizon
            for i, [pred, has_pred] in enumerate(zip(preds, has_preds)):
                if has_pred[frame]:
                    self.plot_sct_traj(pred[frame], i)



if __name__ == "__main__":
    root = "/data1/prediction/dataset/argoverse/traj_gen_preprocess/valid_intermediate/raw/"
    visz = View()

    with open("test.json", "r") as file:
        pred_traj = json.loads(file.readline().strip())
        
        for key, traj in pred_traj.items():

            origin_data = pd.read_pickle(os.path.join(root, f"features_{key}.pkl"))
            graph = origin_data['graph'][0]
            visz.vis_background(graph)