from audioop import bias
import numpy as np
from pr2_utils import *
import math
from numpy import deg2rad
import itertools
# Todo min & max lambda value
# take max out of 9
class slam:
    def __init__(self, lidar, noise, numParticles=5) -> None:
        self.bias = self.biasInit()
        self.numParticles=numParticles
        self.particles = np.zeros((numParticles, 3)) # holds coordinates?
        self.weights = np.ones((numParticles)) / numParticles # holds weights
        self.velNoiseCovar=0#0.005
        self.angNoiseCovar=0#0.0005
        self.logOddsLim = (-100, 100)
        self.MAP=self.initMap()
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
        self.lidar = lidar
        self.lidarPointsInWorld = None
        # self.lidarPointsInWorldList = np.empty((0,2)) # 2 by N
        self.xcar=0
        self.ycar=0
        self.theta=0
        self.pred=[]
        self.bresStack=np.empty((2,0))
        self.lidarInSensor=None
    def biasInit(self):
        bias={}
        for j, (bx, by) in enumerate(itertools.product(range(-4, 5), range(-4, 5))):
            bias[j] = (bx, by)
        return bias
    def predict(self, plot=False): #noise params
        # self.lidarInSensor, a = self.lidar.lidarPointsInVehicle(self.lidar_data[0])
        # self.lidarPointsInWorld = np.array([self.xcar + a[:,0], self.ycar +a[:,1]]).T # 286,2
        # self.lidarPointsInWorld = self.lidarPointsInWorld[self.in_map(self.lidarPointsInWorld), :2]
        self.theta=0
        self.xcar, self.ycar =0, 0
        for i in range(1, len(self.lidar_data[:,0])):

            # if i==20000:
                # break
            # self.differentialDrivePredict(i, 0)
            
            tmp = []
            for p in range(3):
                self.lidarInSensor, a = self.lidar.lidarPointsInVehicle(self.lidar_data[i])
                self.lidarPointsInWorld = np.array([self.particles[p,0] + a[:,0], self.particles[p,0] +a[:,1]]).T # 286,2
                self.lidarPointsInWorld = self.lidarPointsInWorld[self.in_map(self.lidarPointsInWorld), :2]

                self.differentialDrivePredict(i, p)
                self.runBresenham(p) # also updates logodds map
                maxc, idx, particle = self.runMapCorrel(p)
                tmp.append((maxc,self.bias[idx], particle))
                break
            Cstar, istar, particle = max(tmp, key= lambda x: x[0])
            # print("yee",np.sum(self.MAP["mapLogOdds"]))
                # print(self.lidarPointsInWorld.shape)
            # self.resetU()
            print(i)
        
        if plot:
            fig1 = plt.figure()
            plt.imshow(self.MAP['map'],cmap="binary")
            plt.title("Occupancy grid map")
            plt.show(block=True)
            fig2 = plt.figure()
            plt.imshow(self.MAP['map'],cmap="gray")
            plt.title("Occupancy grid map")
            plt.show(block=True)
            fig3 = plt.figure()
            plt.scatter([x[0] for x in self.pred],[y[1] for y in self.pred] )
            plt.show(block=True)
    def differentialDrivePredict(self, i, p)  :
        time_diff1 = (self.encoder_timestamp[i] - self.encoder_timestamp[i-1])*(10**(-9))
        # time_diff2 = (self.fog_timestamp[i] - self.fog_timestamp[i-1])*(10**(-9))
        left_velocity, right_velocity = (self.left_meters[i]-self.left_meters[i-1])/(time_diff1) , (self.right_meters[i]-self.right_meters[i-1])/(time_diff1)
        lin_vel =  (((left_velocity+right_velocity) / 2.0) + noise.gaussNoise(0,self.velNoiseCovar,1)[0]) 
        lin_velNoNoise =  (((left_velocity+right_velocity) / 2.0)) 
        tau = time_diff1

        if p==0:
            self.theta += self.fog_data[i-1][-1]
            self.xcar += (tau*lin_velNoNoise* (math.cos(self.theta)) )
            self.ycar += (tau*lin_velNoNoise* (math.sin(self.theta)) )
            self.pred.append((self.xcar, self.ycar, self.theta))
        
        # self.particles[p,2] += self.fog_data[i-1][-1] + np.random.normal(0, self.velNoiseCovar, 1)[0] # theta
        self.particles[p,2] += tau*(self.fog_data[i-1][-1] + np.random.normal(0, self.velNoiseCovar, 1)[0]/tau) # theta
        self.particles[p,0] += (tau*lin_vel* (math.cos(self.theta)) ) # x
        self.particles[p,1] += (tau*lin_vel* (math.sin(self.theta)) ) # y
        # self.pred.append((self.particles[p,0],self.particles[p,1], self.particles[p,2]))
        # self.pred.append((self.particles[p,0],self.particles[p,1], self.theta))

        # self.pred.append((self.xcar, self.ycar, self.theta))

        
    def runBresenham(self, p):
        for i in range(len(self.lidarPointsInWorld)):
            x,y = self.xcar, self.ycar
            bres = bresenham2D(x,y,self.lidarPointsInWorld[i][0],self.lidarPointsInWorld[i][1])
            self.bresStack=np.hstack((self.bresStack, bres))
            self.updateLogOddsMap(bres)
            # toc(ts, "update")
        self.MAP['mapLogOdds'] = np.clip(self.MAP['mapLogOdds'], *self.logOddsLim) # prevent overconfidence
        # self.in_map(self.bresStack)
        # fig = plt.figure()
        # plt.scatter(bresStack[0], bresStack[1])
        # plt.show(block=True)
    def updateLogOddsMap(self, bres):
        trust = math.log(9)
        # endx,endy = int(bres[0,-1]), int(bres[1,-1])
        endx,endy = self.transformMetersToCells(bres[0,-1], bres[1,-1])
        #LogOdds
        self.MAP['mapLogOdds'][endx,endy] +=trust
        if self.MAP['mapLogOdds'][endx,endy] > 0:
            self.MAP['map'][endx,endy]=1
        # brespixels = self.coordinate_to_index(bres.T)
        # self.MAP["mapLogOdds"][brespixels[:,0],brespixels[:,1]]-=trust
        wut=0
        for i in range(len(bres[0])-1):
            x, y = self.transformMetersToCells(bres[0,i], bres[1,i])
            self.MAP["mapLogOdds"][x,y]-=trust
            # wut-=trust
            if self.MAP['mapLogOdds'][x,y] < 0:
                self.MAP['map'][x,y]=0
        # print("brooo",wut, trust)
        # print("yee",np.sum(self.MAP["mapLogOdds"]))
        
    def resetU(self):
        self.xcar,self.ycar,self.theta=0,0,0
    def initMap(self):
        #res = 1, min_x = -200, max_x = 1300, min_y = -1000 max_y =700
        self.MAP = {}
        self.MAP['res']   = 1 #meters
        self.MAP['xmin']  = -300  #meters
        self.MAP['ymin']  = -1100
        self.MAP['xmax']  =  1400
        self.MAP['ymax']  =  800 
        self.MAP['sizex']  = int(np.ceil((self.MAP['xmax'] - self.MAP['xmin']) / self.MAP['res'] + 1)) #cells
        self.MAP['sizey']  = int(np.ceil((self.MAP['ymax'] - self.MAP['ymin']) / self.MAP['res'] + 1))
        self.MAP['map'] = np.zeros((self.MAP['sizex'],self.MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8 
        # self.MAP['map'] = np.full((self.MAP['sizex'],self.MAP['sizey']),0.8) #DATA TYPE: char or int8 
        self.MAP['mapLogOdds'] = np.zeros((self.MAP['sizex'],self.MAP['sizey'])) #DATA TYPE: char or int8
        self.MAP['mapPMF'] = np.zeros((self.MAP['sizex'],self.MAP['sizey'])) #DATA TYPE: char or int8

        print("MAP, shape: ",self.MAP['map'].shape)
        return self.MAP
    def transformMetersToCells(self,x,y):
        # convert from meters to cells
        xmap = np.ceil((x - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1
        ymap = np.ceil((y - self.MAP['ymin']) / self.MAP['res'] ).astype(np.int16)-1
        return xmap,ymap
    def coordinate_to_index(self, coordinates):
        # coordinates = np.array(coordinates)
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(1, -1)

        tmp= np.hstack([
            np.ceil((coordinates[:, 0] - self.MAP["xmin"]) / self.MAP["res"]).reshape(
                -1, 1),
            np.ceil((coordinates[:, 1] - self.MAP["ymin"]) / self.MAP["res"]).reshape(
                -1, 1),
        ]).astype(np.int32)
        # print(tmp.shape)
        return tmp
    

    def runMapCorrel(self, p):
        x_im = np.arange(self.MAP['xmin'],self.MAP['xmax']+self.MAP['res'],self.MAP['res']) #x-positions of each pixel of the map
        y_im = np.arange(self.MAP['ymin'],self.MAP['ymax']+self.MAP['res'],self.MAP['res']) #y-positions of each pixel of the map
        # self.lidarPointsInWorldList = np.array(self.lidarPointsInWorldList)
        # print(self.lidarPointsInWorldList.shape)
        # Y = self.lidarPointsInWorldList.T
        # Y = self.lidarPointsInWorld
        Y = self.lidarInSensor
        # Y = self.coordinate_to_index(Y)
        increment = 1
        offset = 4
        x_low,x_high = (self.xcar-offset), (self.xcar+offset+increment)
        y_low,y_high = (self.ycar-offset), (self.ycar+offset+increment)
        # x_low,x_high = (-offset), (offset+increment)+increment
        # y_low,y_high = (-offset), (offset+increment)+increment
        xs = np.arange(x_low, x_high, increment)
        ys = np.arange(y_low, y_high, increment)
        c = mapCorrelation(self.MAP['map'],x_im,y_im,Y,xs,ys)
        idx = np.argmax(c)
        cmax = np.max(c)
        print("idx",idx)
        print("bias",self.bias[idx])
        print("cshape",c.shape)
        print(c)
        return (cmax, idx, p)

    def in_map(self, coordinates):
        # coordinates = coordinates.T
        tmp = np.logical_and(
            np.logical_and(self.MAP['xmin'] <= coordinates[:, 0],
                           coordinates[:, 0] <= self.MAP['xmax']),
            np.logical_and(self.MAP['ymin'] <= coordinates[:, 1],
                           coordinates[:, 1] <= self.MAP['ymax']))
        # print("inmap", tmp.shape)
        return tmp
class lidar:
    def __init__(self) -> None:
        self.anglesx = [math.cos(deg2rad(i)) for i in np.arange(-5, 185, 0.666)]
        self.anglesy = [math.sin(deg2rad(i)) for i in np.arange(-5, 185, 0.666)]
        self.lidar2vehicleR = np.array([[0.00130201, 0.796097, 0.605167], [0.999999, -0.000419027, -0.00160026], [-0.00102038, 0.605169, -0.796097] ])
        self.lidar2vehicleP = np.array([0.8349, -0.0126869, 1.76416])
    
    def lidarPointsInVehicle(self, scan):
        #scan is nparray of 286 points from a scan
        # scan = np.array(scan)
        lthresh, hthresh = 5, 60
        # res = (a<3) *b*a
        # res = res[res!=0]
        # thresh = (scan>=lthresh and scan<=hthresh)

        # lidar_x_disp = (scan>=lthresh)*self.anglesx*scan
        # lidar_x_disp = lidar_x_disp[lidar_x_disp!=0]
        # lidar_y_disp = (scan>=lthresh)*self.anglesy*scan
        # lidar_y_disp = lidar_y_disp[lidar_y_disp!=0]
        # lidar_z_disp = (scan>=lthresh)*self.anglesx*scan # doesnt matter can discard
        # lidar_z_disp = lidar_z_disp[lidar_z_disp!=0]

        lidar_x_disp = self.anglesx*scan
        lidar_y_disp = self.anglesy*scan
        lidar_z_disp = 0*scan
        tmp = np.array([lidar_x_disp,lidar_y_disp,lidar_z_disp])
        lidar_vehicle = (self.lidar2vehicleR @ tmp).T + self.lidar2vehicleP
        return np.stack((lidar_x_disp,lidar_y_disp)), lidar_vehicle[:,:-1] # N by 2

class noise():
    def __init__(self) -> None:
        self.vel_covar = 0.5 # encoder
        self.ang_covar = 0.05 # fog
    def gaussNoise(self, mean, covar, particles=1):
        noise = np.random.normal(mean, covar, particles)
        return noise        
if __name__ == '__main__':
    lidar = lidar()
    noise = noise()
    slam = slam(lidar, noise)
    slam.predict(True)

