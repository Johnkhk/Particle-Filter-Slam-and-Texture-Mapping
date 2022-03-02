import numpy as np
from pr2_utils import *

def runMapCorrel(p, x, y, x_im, y_im, lidarInSensor, map, bias, log=False):
    # self.lidarPointsInWorldList = np.array(self.lidarPointsInWorldList)
    # print(self.lidarPointsInWorldList.shape)
    # Y = self.lidarPointsInWorldList.T
    # Y = self.lidarPointsInWorld
    Y = lidarInSensor
    # Y = self.coordinate_to_index(Y)
    increment = 1
    offset = 4
    x_low,x_high = (x-offset), (x+offset+increment)
    y_low,y_high = (y-offset), (y+offset+increment)
    # x_low,x_high = (-offset), (offset+increment)+increment
    # y_low,y_high = (-offset), (offset+increment)+increment
    xs = np.arange(x_low, x_high, increment)
    ys = np.arange(y_low, y_high, increment)
    # c = mapCorrelation(self.MAP['map'],x_im,y_im,Y,xs,ys)
    c = mapCorrelation(map,x_im,y_im,Y,xs,ys)
    idx = np.argmax(c)
    cmax = np.max(c)
    if log:
        print("Yshape",Y.shape)
        print("cshape",c.shape)
        print("idx",idx)
        # print("bias",bias[idx]) # maybe not needed
        print(c)
    return (cmax, idx, p)

def runMapCorrelVectorized(pose, lidarInV, map, slam, log=False):
    x,y,theta = pose[0],pose[1],pose[2]
    grid = map.copy()
    lidarInW = np.array([x,theta]) + lidarInV
    lidarInW = Rotate2D(theta, lidarInW)
    lidarInM = slam.mapLogOdds.coordToMap(lidarInW)
    uh = map[lidarInM[:,0],lidarInM[:,1]]
    print(uh.shape)

def Rotate2D(theta, arr):
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    if arr.shape[0]==2:
        return R@arr
    else:
        return (R@(arr.T)).T

def getStereoDisparity(path_l, path_r):
    image_l = cv2.imread(path_l, 0)
    image_r = cv2.imread(path_r, 0)
    image_l_gray = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
    image_r_gray = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)

    # You may need to fine-tune the variables `numDisparities` and `blockSize` based on the desired accuracy
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=9) 
    disparity = stereo.compute(image_l_gray, image_r_gray)
    return disparity