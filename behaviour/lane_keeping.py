from .base import Behaviour
from argoverse.map_representation.map_api import ArgoverseMap

class LaneKeeping(Behaviour):
    def __init__(self) -> None:
        super().__init__()
        self.am = ArgoverseMap()
    
    def behave(self, trajs, city_name) -> bool:
        x = []
        for traj in trajs:
            closest_lane_obj, conf, dense_centerline = self.am.get_nearest_centerline(traj, city_name)
            x.append(closest_lane_obj.id)
        length = len(x)
        for i in range(length - 1):
            if x[i] != x[i+1]:
                successor_ids = self.am.get_lane_segment_successor_ids(x[i], city_name)
                processor_ids = self.am.get_lane_segment_predecessor_ids(x[i], city_name)
                if not (x[i+1] in successor_ids or x[i+1] in processor_ids):
                   
                    return False
        return True