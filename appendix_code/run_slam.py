from data import *
from slam import *
from pr2_utils import *
from lidar import *

class run_slam():
    def __init__(self, slam, lidar) -> None:
        pkl_file = open('data.pkl', 'rb')
        data = pickle.load(pkl_file)
        slam = slam(5, (-300, 1400), (-1100, 800), 1)
        self.theta, self.x_pred, self.y_pred = 0, 0, 0
        self.noiseSigma = np.diag([5e-2, 5e-2, 5e-2])
        lidar = lidar(slam.mapLogOdds)
    def run():
        for i in range(1, len(data.lidar_data[:,0])):
            time_diff1 = (data.encoder_timestamp[i] - data.encoder_timestamp[i-1])*(10**(-9))
            time_diff2 = (data.fog_timestamp[i] - data.fog_timestamp[i-1])*(10**(-9))
            left_velocity, right_velocity = (data.left_meters[i]-data.left_meters[i-1])/(time_diff1) , (data.right_meters[i]-data.right_meters[i-1])/(time_diff1)
            lin_vel =  (((left_velocity+right_velocity) / 2.0))
            tau = time_diff1
            self.slam.theta += tau* (data.fog_data[i-1][-1]  / tau)
            self.slam.x_pred += (tau*lin_vel* (math.cos(self.theta)) )
            self.slam.y_pred += (tau*lin_vel* (math.sin(self.theta)) )

            lidarInWorld = lidar.getLidarInWorld(data.lidar_data[i], x_pred, y_pred)

            u = np.array([self.x_pred, self.y_pred, self.theta])
            u = np.random.multivariate_normal(np.array(u), self.noiseSigma, 5)
            # print(u.shape)
            # print(u)

            # slam.predict(u)
            print(i)