from os import name

from pip import main
from pr2_utils import *
import math
import pickle

class data:
    def __init__(self) -> None:
        self.fog_timestamp, self.fog_data = read_data_from_csv("data/sensor_data/fog.csv")
        self.fog_timestamp = np.array([sum(self.fog_timestamp[i:i+10]) for i in range(0, len(self.fog_timestamp), 10)])
        self.fog_data = np.array([np.sum(self.fog_data[i:i+10], axis=0) for i in range(0, len(self.fog_data), 10)])
        self.encoder_timestamp, self.encoder_data = read_data_from_csv("data/sensor_data/encoder.csv")
        self.encoder_resolution = 4096   
        self.encoder_left_wheel_diameter = 0.623479
        self.encoder_right_wheel_diameter =  0.622806
        self.encoder_wheel_base = 1.52439
        self.left_meters = math.pi * (self.encoder_left_wheel_diameter) * self.encoder_data[:,0] / self.encoder_resolution 
        self.right_meters = math.pi * (self.encoder_right_wheel_diameter) * self.encoder_data[:,1] / self.encoder_resolution
        self.lidar_timestamp, self.lidar_data = read_data_from_csv("data/sensor_data/lidar.csv")
        # self.lidar_data=np.clip(np.array(self.lidar_data),5,60)
        self.lidar_data=np.array(self.lidar_data)
if __name__ == '__main__':
    obj = data()
    output = open('data.pkl', 'wb')
    pickle.dump(obj, output)
