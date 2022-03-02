import numpy as np
from pr2_utils import *
import math
from numpy import deg2rad
# Todo min & max lambda value
# take max out of 9
class slam:
    def __init__(self, lidar, noise, numParticles=5) -> None:
        self.numParticles=numParticles
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
        self.lidar_data=np.clip(np.array(self.lidar_data),5,60)
        self.lidar = lidar
        self.lidarPointsInWorld = None
        # self.lidarPointsInWorldList = np.empty((0,2)) # 2 by N
        self.xcar=0
        self.ycar=0
        self.bresStack=np.empty((2,0))
        self.pred=[]

    def predict(self, plot=False): #noise params
        self.pred = []
        x_pred, y_pred =0, 0
        # pred.append((-0.335,-0.035))
        theta=0
        self.pred.append((x_pred,y_pred,theta))

        # for p in range(self.numParticles):
        for p in range(1):
            x_pred, y_pred =0, 0
            self.xcar,self.ycar = x_pred,y_pred
            theta=0
            for i in range(1, len(self.lidar_data[:,0])):
                time_diff1 = (self.encoder_timestamp[i] - self.encoder_timestamp[i-1])*(10**(-9))
                time_diff2 = (self.fog_timestamp[i] - self.fog_timestamp[i-1])*(10**(-9))
                left_velocity, right_velocity = (self.left_meters[i]-self.left_meters[i-1])/(time_diff1) , (self.right_meters[i]-self.right_meters[i-1])/(time_diff1)
                lin_vel =  (((left_velocity+right_velocity) / 2.0) + noise.gaussNoise(0,self.velNoiseCovar,1)[0]) 
                # Vehicle to FOG: -0.335 -0.035 0.78
                tau = time_diff1
                theta += tau* ( (self.fog_data[i-1][-1] + noise.gaussNoise(0,self.angNoiseCovar,1)[0]) / tau)
                x_pred += (tau*lin_vel* (math.cos(theta)) )
                y_pred += (tau*lin_vel* (math.sin(theta)) )
                self.xcar,self.ycar = x_pred,y_pred
                if i%100!=0 or i==1:
                    continue
                self.pred.append((x_pred,y_pred, theta))
                #Lidar stuff
                a = self.lidar.lidarPointsInVehicle(self.lidar_data[i])
                self.lidarPointsInWorld = np.array([x_pred + a[:,0], y_pred +a[:,1]]).T # 286,2
                # self.lidarPointsInWorldList = np.vstack((self.lidarPointsInWorldList,self.lidarPointsInWorld))
                #just the lidar scan
                # plt.plot(self.lidarPointsInWorld[:,0], self.lidarPointsInWorld[:,1])
                # plt.show(block=True)
                # break
                # ts = tic()
                self.runBresenham(x_pred,y_pred)
                # self.runMapCorrel()
                # break
                print(i)
                # if i == 5000:
                    # break
                # break
            # fig2 = plt.figure()
            # plt.imshow(self.MAP['map'],cmap="hot")
            # plt.imshow(self.MAP['mapLogOdds'],cmap="hot")
            # plt.title("Occupancy grid map")
            # plt.show(block=True)
        # self.runMapCorrel()
        # self.MAP['map'][self.MAP["mapLogOdds"]>0]=1

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

            # fig, ax = plt.subplots(figsize=(20, 20), dpi=80)
            fig4 = plt.figure()
            self.MAP['mapLogOdds'] = np.clip(self.MAP['mapLogOdds'], *self.logOddsLim)
            gamma = 1.0 - 1.0/(1.0 + np.exp(self.MAP['mapLogOdds']))
            plt.imshow(gamma,cmap="gray")
            # ax.imshow(gamma,cmap="bone")
            # for px, py, _ in self.pred:
            #     px, py = self.transformMetersToCells(px,py)
            #     ax.plot(px, py, marker='o', color='#ff4733', ms=1)

            plt.show(block=True)
        
    def runBresenham(self, x_pred, y_pred):
        self.bresStack=np.empty((2,0))
        for i in range(len(self.lidarPointsInWorld)):
            # print("YOOOOO", x_pred, self.lidarPointsInWorld[i][0], y_pred, self.lidarPointsInWorld[i][1])
            bres = bresenham2D(x_pred,y_pred,self.lidarPointsInWorld[i][0],self.lidarPointsInWorld[i][1])
            # print(bres.shape)
            self.bresStack=np.hstack((self.bresStack, bres))
            # print("PPP"*60)
            # ts = tic()
            self.updateLogOddsMap(bres, x_pred, y_pred)
            # toc(ts, "update")
        # fig = plt.figure()
        # plt.plot(self.bresStack[0], self.bresStack[1])
        # plt.show(block=True)
        
    def plotBres(self,bres):
        pass
    def initMap(self):
        #res = 1, min_x = -200, max_x = 1300, min_y = -1000 max_y =700
        self.MAP = {}
        self.MAP['res']   = 1 #meters
        self.MAP['xmin']  = -200  #meters
        self.MAP['ymin']  = -1000
        self.MAP['xmax']  =  1300
        self.MAP['ymax']  =  700 
        self.MAP['sizex']  = int(np.ceil((self.MAP['xmax'] - self.MAP['xmin']) / self.MAP['res'] + 1)) #cells
        self.MAP['sizey']  = int(np.ceil((self.MAP['ymax'] - self.MAP['ymin']) / self.MAP['res'] + 1))
        self.MAP['map'] = np.zeros((self.MAP['sizex'],self.MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8 
        # self.MAP['map'] = np.full((self.MAP['sizex'],self.MAP['sizey']),0.8) #DATA TYPE: char or int8 
        self.MAP['mapLogOdds'] = np.zeros((self.MAP['sizex'],self.MAP['sizey'])) #DATA TYPE: char or int8
        self.MAP['mapPMF'] = np.zeros((self.MAP['sizex'],self.MAP['sizey'])) #DATA TYPE: char or int8

        print("MAP, shape: ",self.MAP['map'].shape)
        return self.MAP
    # def updateLogOdds(self, bresenham):
    #     trust = math.log(9)
    #     for i in range(len(bresenham[0])-1):
    #         x,y = int(bresenham[0,i]), int(bresenham[1,i])
    #         self.MAP['map'][x+100,y] -= trust
    #     endx,endy = int(bresenham[0,-1]), int(bresenham[1,-1])
    #     self.MAP['map'][endx,endy] += trust
    #     pass
    def updateMap(self):
        for i in range(len(self.lidarPointsInWorld)):
            xs0,ys0 = self.lidarPointsInWorld[i]
            xis = np.ceil((xs0 - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1
            yis = np.ceil((ys0 - self.MAP['ymin']) / self.MAP['res'] ).astype(np.int16)-1
            # print(xis,yis)
            self.MAP['map'][xis,yis] = 1
    def updateMapb(self):
        for i in range(len(self.lidarPointsInWorld)):
            xs0,ys0 = self.lidarPointsInWorld[i]
            xis = np.ceil((xs0 - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1
            yis = np.ceil((ys0 - self.MAP['ymin']) / self.MAP['res'] ).astype(np.int16)-1
            # print(xis,yis)
            self.MAP['map'][xis,yis] = 1
        # fig2 = plt.figure()
        # plt.imshow(self.MAP['map'],cmap="hot")
        # plt.title("Occupancy grid map")
        # plt.show(block=True)  

    def transformMetersToCells(self,x,y):
        # convert from meters to cells
        xmap = np.ceil((x - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1
        ymap = np.ceil((y - self.MAP['ymin']) / self.MAP['res'] ).astype(np.int16)-1
        return xmap,ymap
    def updateLogOddsMap(self, bres, x_pred, y_pred):
        trust = math.log(9)
        # endx,endy = int(bres[0,-1]), int(bres[1,-1])
        endx,endy = self.transformMetersToCells(bres[0,-1], bres[1,-1])
        #LogOdds
        self.MAP['mapLogOdds'][endx,endy] +=trust
        if self.MAP['mapLogOdds'][endx,endy] > 0:
            self.MAP['map'][endx,endy]=1
        # self.MAP["mapLogOdds"][startx:endx,starty:endy]-=trust
        # self.MAP["mapLogOdds"][startx:endx,starty:endy]-=trust
        for i in range(len(bres)-1):
            x, y = self.transformMetersToCells(bres[0,i], bres[1,i])
            self.MAP["mapLogOdds"][x,y]-=trust
            if self.MAP['mapLogOdds'][x,y] < 0:
                self.MAP['map'][x,y]=0
            # self.MAP["mapLogOdds"][int(bres[0,i]),int(bres[1,i])]-=trust
            # if self.MAP['mapLogOdds'][int(bres[0,i]),int(bres[1,i])] < 0:
            #     self.MAP['map'][int(bres[0,i]),int(bres[1,i])]=0
            
        #update occupancy grid
        # ts=tic()
        # self.MAP['map'][self.MAP["mapLogOdds"]>0]=1
        # toc(ts, "ting")

        #SeeMap of rays
        # self.MAP['mapLogOdds'][endx,endy] =1
        # self.MAP["mapLogOdds"][startx:endx,starty:endy]=1
    def runMapCorrel(self):
        x_im = np.arange(self.MAP['xmin'],self.MAP['xmax']+self.MAP['res'],self.MAP['res']) #x-positions of each pixel of the map
        y_im = np.arange(self.MAP['ymin'],self.MAP['ymax']+self.MAP['res'],self.MAP['res']) #y-positions of each pixel of the map
        # self.lidarPointsInWorldList = np.array(self.lidarPointsInWorldList)
        # print(self.lidarPointsInWorldList.shape)
        # Y = self.lidarPointsInWorldList.T
        Y = self.lidarPointsInWorld
        increment = 1
        offset = 4
        x_low,x_high = (self.xcar-offset), (self.xcar+offset+increment)+increment
        y_low,y_high = (self.ycar-offset), (self.ycar+offset+increment)+increment
        xs = np.arange(x_low, x_high, increment)
        ys = np.arange(y_low, y_high, increment)
        c = mapCorrelation(self.MAP['map'],x_im,y_im,Y,xs,ys)
        print(c.shape)
        print(c)


class lidar:
    def __init__(self) -> None:
        self.anglesx = [math.cos(deg2rad(i)) for i in np.arange(-5, 185, 0.666)]
        self.anglesy = [math.sin(deg2rad(i)) for i in np.arange(-5, 185, 0.666)]
        self.lidar2vehicleR = np.array([[0.00130201, 0.796097, 0.605167], [0.999999, -0.000419027, -0.00160026], [-0.00102038, 0.605169, -0.796097] ])
        self.lidar2vehicleP = np.array([0.8349, -0.0126869, 1.76416])
    
    def lidarPointsInVehicle(self, scan):
        #scan is nparray of 286 points from a scan
        # scan = np.array(scan)
        lidar_x_disp = self.anglesx*scan
        lidar_y_disp = self.anglesy*scan
        lidar_z_disp = 0*scan
        tmp = np.array([lidar_x_disp,lidar_y_disp,lidar_z_disp])
        lidar_vehicle = (self.lidar2vehicleR @ tmp).T + self.lidar2vehicleP
        return lidar_vehicle[:,:-1] # N by 2

# class particles:
#     def __init__(self, numParticles, noise) -> None:
#         self.noise = noise
#         self.numParticles = numParticles
#         self.particles=self.initParticles(self.numParticles)

#     def initParticle(self, mean, covar):
#         pass
#     def gaussNoise():

#         pass
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

