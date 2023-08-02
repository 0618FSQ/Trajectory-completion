from matplotlib import pyplot as plt
import json
import os
import numpy as np
import pandas as pd
import warnings
from argoverse.utils.mpl_plotting_utils import visualize_centerline
from tqdm import tqdm

warnings.filterwarnings("ignore")

RESCALE_LENGTH = 1.0    # the rescale length th turn the lane vector into equal distance pieces


class View():
    def __init__(self,):

        self.LANE_WIDTH = {'MIA': 3.84, 'PIT': 3.97}
        self.COLOR_DICT = {"AGENT": "red", "OTHERS": "green", "AV": "yellow"}

    def view_data(self, data):
        fig = plt.figure(0, figsize=(8, 7))
        fig.clear()

        # visualize the centerlines
        lines_ctrs = data['graph']['ctrs']
        lines_feats = data['graph']['feats']
        lane_idcs = data['graph']['lane_idcs']
        for i in np.unique(lane_idcs):
            line_ctr = lines_ctrs[lane_idcs == i]
            line_feat = lines_feats[lane_idcs == i]
            line_str = (2.0 * line_ctr - line_feat) / 2.0
            line_end = (2.0 * line_ctr[-1, :] + line_feat[-1, :]) / 2.0
            line = np.vstack([line_str, line_end.reshape(-1, 2)])
            visualize_centerline(line)

        # visualize the trajectory
        trajs = data['feats'][:, :, :2]
        has_obss = data['has_obss']
        preds = data['gt_preds']
        has_preds = data['has_preds']
        for i, [traj, has_obs, pred, has_pred] in enumerate(zip(trajs, has_obss, preds, has_preds)):
            self.plot_traj(traj[has_obs], pred[has_pred], i)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        # plt.savefig("./1.jpg")
        plt.show()

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
            plt.plot(lineX, lineY, "-", color="grey", alpha=1, linewidth=0.5, zorder=0)
            # plt.text(lineX[0], lineY[0], "s")
            # plt.text(lineX[-1], lineY[-1], "e")
            plt.axis("equal")

    def visualize_result(self, data, preds):
        pass

    def visualize_data(self, data, preds, save_path , idx):
        """
        visualize the extracted data, and exam the data
        """
        fig = plt.figure(0, figsize=(8, 7), dpi=500)
        fig.clear()

        self.vis_background(data['graph'][0])

        # visualize the trajectory
        trajs = data['agt_traj'][0]
        box_min = origin_data['box_min'].values[0]
        box_max = origin_data['box_max'].values[0]
        trajs = trajs * (box_max - box_min) + box_min

        # preds = data['gt_preds'][0]

        self.plot_traj(trajs, preds, 0)
        
        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.savefig(os.path.join(save_path, (str(idx) + ".jpg")))
        plt.show()
        
        # plt.show(block=False)
        # plt.pause(0.5)

    def plot_reference_centerlines(self, cline_list, splines, obs, pred, ref_line_idx):
        fig = plt.figure(0, figsize=(8, 7))
        fig.clear()

        for centerline_coords in cline_list:
            visualize_centerline(centerline_coords)

        for i, spline in enumerate(splines):
            xy = np.stack([spline.x_fine, spline.y_fine], axis=1)
            if i == ref_line_idx:
                plt.plot(xy[:, 0], xy[:, 1], "--", color="r", alpha=0.5, linewidth=1, zorder=10)
            else:
                plt.plot(xy[:, 0], xy[:, 1], "--", color="b", alpha=0.5, linewidth=1, zorder=10)

        self.plot_traj(obs, pred)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.savefig('./1.png')
        plt.show()
        # plt.show(block=False)
        # plt.pause(0.5)

    def plot_traj(self, obs, pred, traj_id=None):
        assert len(obs) != 0, "ERROR: The input trajectory is empty!"
        traj_na = "t{}".format(traj_id) if traj_id else "traj"
        obj_type = "AGENT" if traj_id == 0 else "OTHERS"

        if obj_type == 'AGENT':
            plt.plot(obs[:, 0], obs[:, 1], color='b', alpha=1, linewidth=0.5, zorder=15)

            plt.plot(pred[:, 0], pred[:, 1], color='red', alpha=1, linewidth=0.5, zorder=15)
        # else:
        #     plt.plot(obs[:, 0], obs[:, 1], color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)
        #     plt.plot(pred[:, 0], pred[:, 1], color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)
        
        # plt.text(obs[0, 0], obs[0, 1], "{}_s".format(traj_na))


if __name__ == "__main__":

    root = "/data1/prediction/dataset/argoverse/traj_gen_preprocess/valid_intermediate/raw/"
    save_path = "picture"
    visz = View()
    
    with open("trajectories2/restart.json", "r") as file:
        pred_traj = json.loads(file.readline().strip())
        for i, (key, traj) in enumerate(tqdm(pred_traj.items())):
            if i < 21:
                continue
            assert (np.asarray(traj).shape == (50,2)), "shape error"
            
            origin_data = pd.read_pickle(os.path.join(root, f"features_{key}.pkl"))
            box_min = origin_data['box_min'].values[0]
            box_max = origin_data['box_max'].values[0]
            
            real_traj = traj * (box_max - box_min) + box_min
            box_max = origin_data['box_max'].values[0]

            real_traj = traj * (box_max - box_min) + box_min
            
            graph = origin_data['graph'][0]
            visz.visualize_data(origin_data, real_traj, save_path, i)
            if i == 50:
                break