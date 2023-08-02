from collections import defaultdict
import numpy as np
from argoverse.map_representation.map_api import ArgoverseMap
from pyclothoids import SolveG2
from util.calculator import Calculator
import networkx as nx
from util.graph import Graph
import numpy as np



class PostPreprocessor:
    def __init__(
        self, 
        v_x_max=20, 
        v_x_std=1.2,
        v_y_std=0.5,
        a_x_std=1.2,
        a_y_std=0.5,
        v_x_min=8, 
        v_y_max=1.0, 
        v_y_min=-1.0, 
        acc_x_max=2,
        acc_x_min=-5, 
        acc_y_min=-0.1,
        acc_y_max=0.3, 
        heading_delta=0.2
    ):
        self.am = ArgoverseMap()
        self.v_x_max = v_x_max
        self.v_y_max = v_y_max
        self.v_x_min = v_x_min
        self.v_y_min = v_y_min
        self.acc_x_max = acc_x_max
        self.acc_y_min = acc_y_min
        self.acc_x_min = acc_x_min
        self.acc_y_max = acc_y_max
        self.heading_delta = heading_delta

        self.v_x_std=v_x_std
        self.v_y_std=v_y_std
        self.a_x_std=a_x_std
        self.a_y_std=a_y_std

    def postpreprocess(
        self, 
        traj:np.ndarray, 
        rot:np.ndarray, 
        orig:np.ndarray, 
        box_min:np.ndarray, 
        box_max:np.ndarray
    ):
        real_traj = traj * (box_max - box_min) + box_min
        real_traj = np.matmul(np.linalg.inv(rot), real_traj.T).T + orig
        return real_traj
    
    def permutate(self, real_traj, city_name):
        x = []
        cache = defaultdict(list)
        for traj in real_traj:
            closest_lane_obj, conf, dense_centerline = self.am.get_nearest_centerline(traj, city_name)
            if conf < 0.85:
                return []
            x.append(closest_lane_obj.id)
            cache[closest_lane_obj.id].append(
                traj.tolist()
            )
        set_x = set(x)
        edges = []
        for i in x:
            successor_ids = self.am.get_lane_segment_successor_ids(i, city_name) # 后继
            left_and_right_ids = self.am.get_lane_segment_adjacent_ids(i, city_name)
            edges += [(i, j) for j in successor_ids if j in set_x]
            edges += [(i, j) for j in left_and_right_ids if j is not None and j in set_x]
        graph = Graph(nodes=set_x, edges=list(set(edges)), is_bi=False)
        nodes = graph.max_connected_nodes()
        new_edges = [edge for edge in set(edges) if set(edge).issubset(nodes)]

        bi_graph = Graph(nodes=nodes, edges=new_edges, is_bi=True)

        start_node = bi_graph.get_in_degree_zero()
        end_node = bi_graph.get_out_degree_zero()
        new_ids = []
        if len(start_node) > 1 and len(end_node) > 1:
            return []
        else:
            if len(start_node) == 1:
                p = start_node[0]
                while p != end_node[0]:
                    new_ids.append(p)
                    next_node = bi_graph.nerghbor(p)
                    if len(next_node) == 1:
                        p = next_node[0]
                    elif len(next_node) == 2:
                        p1, p2 = next_node
                        if x.index(p1) < x.index(p2):
                            p = p1
                        else:
                            p = p2
                    else:
                        return []
                new_ids += end_node
        new_traj = []
        new_cache = {}
        for new_id in new_ids:
            all_traj = cache[new_id]
            drection, conf = self.am.get_lane_direction(all_traj[0], city_name)
            heading = np.arctan2(*drection.tolist())
            if - np.pi / 2 <= heading <= np.pi / 2:
                all_traj.sort()
            else:
                all_traj.sort(reverse=True)
            new_cache[new_id] = all_traj
            new_traj += all_traj
        if len(new_traj) == 0:
            return []
        new_traj = self.constrain(cache=new_cache, traj=new_traj, city_name=city_name)
        return new_traj
    
    def cross(self, real_traj, city_name):
        cache = defaultdict(bool)
        keys = []
        for traj in real_traj:
            closest_lane_obj, conf, dense_centerline = self.am.get_nearest_centerline(traj, city_name)
            if closest_lane_obj is None:
                return False
            if not cache[closest_lane_obj.id]:
                cache[closest_lane_obj.id] = True
                keys.append(closest_lane_obj.id)
        if len(keys) == 1:
            return False
        length = len(keys)
        def helper(start):
            if start >= length - 1:
                return True
            else:
                boundary_ids = self.am.get_lane_segment_adjacent_ids(keys[start], city_name)
                if keys[start+1] in boundary_ids:
                    return helper(start+1)
                else:
                    return False
        return helper(0)

    def constrain(self, cache, traj, city_name):
        lane_ids = list(cache.keys())
        length = len(lane_ids)
        trajectory = [traj[0]]
        point = 0
        current_lane_id = lane_ids[point]
        calculator = Calculator(x=np.array(traj)[:, 0], y = np.array(traj)[:, 0], time=len(traj) / 10)
        state_v_x, state_v_y, state_acc_x, state_acc_y = [], [], [], []
        heading = calculator.heading
        index = 0
        while len(trajectory) != 50:
            current_location = trajectory[-1]
            while True:
                if index == 0:
                    drection, conf = self.am.get_lane_direction(trajectory[-1], city_name)
                    if not self.am.lane_is_in_intersection(current_lane_id, city_name):
                        delta = np.random.uniform(-0.01, 0.01)
                    else:
                        delta = np.random.uniform(-0.1, 0.1)
                    heading2 = np.arctan2(*drection.tolist())
                    if abs(heading[0] - heading2) > np.pi / 6:
                        new_head = heading2 + delta
                    else:
                        new_head = min(heading2 + delta, heading[0])
                    v_x = np.random.uniform(self.v_x_min, self.v_x_max)
                    v_y = np.random.uniform(self.v_y_min, self.v_y_max)
                    a_x = np.random.uniform(self.acc_x_min, self.acc_x_max)
                    a_y = np.random.uniform(self.acc_y_min, self.acc_y_max)
                    
                    new_position = [
                        current_location[0] + (v_x*np.cos(new_head) - v_y*np.sin(new_head))*0.1 + 0.5*(a_x*np.cos(new_head) - a_y*np.sin(new_head))*0.1**2 , 
                        current_location[1] + (v_y*np.cos(new_head) + v_x*np.sin(new_head))*0.1 + 0.5*(a_y*np.cos(new_head) + a_x*np.sin(new_head))*0.1**2
                    ]
                    
                    closest_lane_obj, conf, dense_centerline = self.am.get_nearest_centerline(new_position, city_name=city_name)
                    if conf < 0.85:
                        continue
                    elif closest_lane_obj.id in lane_ids[point: point + 2]:
                        trajectory.append(new_position)
                        state_v_x.append(v_x)
                        state_v_y.append(v_y)
                        state_acc_x.append(a_x)
                        state_acc_y.append(a_y)
                        if point != length - 1 and closest_lane_obj.id == lane_ids[point+1]:
                            point += 1
                        break
                    else:
                        continue
                else:
                    drection, conf = self.am.get_lane_direction(trajectory[-1], city_name)
                    if not self.am.lane_is_in_intersection(current_lane_id, city_name):
                        delta = np.random.uniform(-0.01, 0.01)
                    else:
                        delta = np.random.uniform(-0.1, 0.1)
                    heading2 = np.arctan2(*drection.tolist())
                    if abs(heading[0] - heading2) > np.pi / 6:
                        new_head = heading2 + delta
                    else:
                        new_head = min(heading2 + delta, heading[0])
                    if np.random.uniform(0, 1) > 0.9:
                        v_x = np.random.uniform(state_v_x[-1]+state_acc_x[-1]*0.1, state_v_x[-1]-state_acc_x[-1]*0.1)
                        v_y = np.random.uniform(state_v_y[-1]+state_acc_y[-1]*0.1, state_v_y[-1]-state_acc_y[-1]*0.1)
                        a_x = np.random.uniform(self.acc_x_min, self.acc_x_max)
                        a_y = np.random.uniform(self.acc_y_min, self.acc_y_max)
                    else:
                        v_x = np.random.uniform(
                            min(
                                state_v_x[-1]+self.v_x_std, 
                                self.v_x_max
                            ), 
                            max(
                                state_v_x[-1]-self.v_x_std, 
                                self.v_x_min
                            )
                        )
                        v_y = np.random.uniform(
                            min(
                                state_v_y[-1]+self.v_y_std, 
                                self.v_y_max
                            ), 
                            max(
                                state_v_y[-1]-self.v_y_std, 
                                self.v_y_min
                            )
                            
                        )
                        a_x = np.random.uniform(state_acc_x[-1] + self.a_x_std, state_acc_x[-1] - self.a_x_std)
                        a_y = np.random.uniform(state_acc_y + self.a_y_std, state_acc_y - self.a_y_std)
                    new_position = [
                        current_location[0] + (v_x*np.cos(new_head) - v_y*np.sin(new_head))*0.1 + 0.5*(a_x*np.cos(new_head) - a_y*np.sin(new_head))*0.1**2 , 
                        current_location[1] + (v_y*np.cos(new_head) + v_x*np.sin(new_head))*0.1 + 0.5*(a_y*np.cos(new_head) + a_x*np.sin(new_head))*0.1**2
                    ]
                    closest_lane_obj, conf, dense_centerline = self.am.get_nearest_centerline(new_position, city_name=city_name)
                    if conf < 0.85:
                        continue
                    elif closest_lane_obj.id in lane_ids[point: point + 2]:
                        trajectory.append(new_position)
                        state_v_x.append(v_x)
                        state_v_y.append(v_y)
                        state_acc_x.append(a_x)
                        state_acc_y.append(a_y)
                        if point != length - 1 and closest_lane_obj.id == lane_ids[point+1]:
                            point += 1
                        break
                    else:
                        continue
        return trajectory
