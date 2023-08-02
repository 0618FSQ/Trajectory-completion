from behaviour.lane_keeping import LaneKeeping

if __name__ == "__main__":
    lane_keeping = LaneKeeping()
    import pandas as pd
    path = "/home/dataset/argoverse/csv/train/16670.csv"
    csv_data = pd.read_csv(path)
    trajs = csv_data[csv_data['OBJECT_TYPE']=="AGENT"][['X', 'Y']].values
    city_name = csv_data['CITY_NAME'].values[0]
    lane_keeping.behave(trajs, city_name)