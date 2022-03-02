import math
import numpy as np
from numpy import deg2rad
# from code.data import data
class lidar:
    def __init__(self, map) -> None:
        self.anglesx = [math.cos(deg2rad(i)) for i in np.arange(-5, 185, 0.666)]
        self.anglesy = [math.sin(deg2rad(i)) for i in np.arange(-5, 185, 0.666)]
        self.lidar2vehicleR = np.array([[0.00130201, 0.796097, 0.605167], [0.999999, -0.000419027, -0.00160026], [-0.00102038, 0.605169, -0.796097] ])
        self.lidar2vehicleP = np.array([0.8349, -0.0126869, 1.76416])

        self.lidarInSensor = None
        self.lidarPointsInWorld = None
        self.MAP = map
    
    def lidarPointsInVehicle(self, scan):
        #scan is nparray of 286 points from a scan
        # scan = np.array(scan)
        lthresh, hthresh = 5, 60
        

        lidar_x_disp = self.anglesx*scan
        lidar_y_disp = self.anglesy*scan
        lidar_z_disp = 0*scan
        tmp = np.array([lidar_x_disp,lidar_y_disp,lidar_z_disp])
        lidar_vehicle = (self.lidar2vehicleR @ tmp).T + self.lidar2vehicleP
        return np.stack((lidar_x_disp,lidar_y_disp)), lidar_vehicle[:,:-1] # N by 2
    def getLidarInWorld(self, scan, x, y):

        self.lidarInSensor, a = self.lidarPointsInVehicle(scan) #self.lidar_data[i]
        # self.lidarPointsInWorld = np.array([self.particles[p,0] + a[:,0], self.particles[p,0] +a[:,1]]).T # 286,2
        self.lidarPointsInWorld = np.array([x + a[:,0], y +a[:,1]]).T # 286,2
        # self.lidarPointsInWorld = self.lidarPointsInWorld[self.MAP.in_map(self.lidarPointsInWorld), :2]
        self.lidarPointsInWorld = self.lidarPointsInWorld[self.MAP.in_map(self.lidarPointsInWorld)]
    def getLidarInV(self, scan):

        self.lidarInSensor, a = self.lidarPointsInVehicle(scan) #self.lidar_data[i]
        # self.lidarPointsInWorld = np.array([self.particles[p,0] + a[:,0], self.particles[p,0] +a[:,1]]).T # 286,2
        self.lidarPointsInWorld = np.array([a[:,0], a[:,1]]).T # 286,2
        # self.lidarPointsInWorld = self.lidarPointsInWorld[self.MAP.in_map(self.lidarPointsInWorld), :2]
        self.lidarPointsInWorld = self.lidarPointsInWorld[self.MAP.in_map(self.lidarPointsInWorld)]

        return self.lidarPointsInWorld, self.lidarInSensor
    # def getLidarInWorldForParticles(self, scan, particles):

    #     self.lidarInSensor, a = self.lidarPointsInVehicle(scan) #self.lidar_data[i]
    #     # self.lidarPointsInWorld = np.array([self.particles[p,0] + a[:,0], self.particles[p,0] +a[:,1]]).T # 286,2
    #     self.lidarPointsInWorldParticles = np.array([x + a[:,0], y +a[:,1]]).T # 286,2
    #     self.lidarPointsInWorldParticles = self.lidarPointsInWorldParticles[self.MAP.in_map(self.lidarPointsInWorld), :2]
    #     return self.lidarPointsInWorldParticles
    def getLidarPointsInVwithz(self, scan):
        lthresh, hthresh = 5, 60
        lidar_x_disp = self.anglesx*scan
        lidar_y_disp = self.anglesy*scan
        lidar_z_disp = 0*scan
        tmp = np.array([lidar_x_disp,lidar_y_disp,lidar_z_disp])
        lidar_vehicle = (self.lidar2vehicleR @ tmp).T + self.lidar2vehicleP
        return lidar_vehicle