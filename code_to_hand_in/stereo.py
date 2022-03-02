
import numpy as np
from numpy.linalg import inv
import cv2
class stereo:
    def __init__(self) -> None:
        self.imagesize = (1280, 560)
        # stereo to vehicle Transforms
        self.StereoToV_R = np.array([[-0.00680499, -0.0153215, 0.99985], [-0.999977, 0.000334627, -0.00680066], [-0.000230383, -0.999883, -0.0153234]])
        self.StereoToV_T = np.array([1.64239, 0.247401, 1.58411])
        # Camera Intrinsics (projection) #3 by 4, K is 3 by 3
        self.l_camera_intrinsicMat = np.array([ [7.7537235550066748e+02, 0., 6.1947309112548828e+02, 0.], [0., 7.7537235550066748e+02, 2.5718049049377441e+02, 0.], [0., 0., 1.,0.] ])
        self.r_camera_intrinsicMat = np.array([ [7.7537235550066748e+02, 0., 6.1947309112548828e+02,-3.6841758740842312e+02], [0., 7.7537235550066748e+02, 2.5718049049377441e+02, 0.], [0., 0., 1., 0.] ])
        # pixel scaling (Ks)
        self.l_camera_matrix = np.array([ 8.1690378992770002e+02, 5.0510166700000003e-01,6.0850726281690004e+02, 0., 8.1156803828490001e+02,2.6347599764440002e+02, 0., 0., 1. ])
        self.r_camera_matrix = np.array([ 8.1378205539589999e+02, 3.4880336220000002e-01,6.1386419539320002e+02, 0., 8.0852165574269998e+02,2.4941049348650000e+02, 0., 0., 1. ])
        #focal length slide 31 lec 8 
        self.oRr = np.array([[0,-1,0],[0,0,-1],[1,0,0]])

        self.baseline = 475.143600050775e-3 # meters
        self.fsu = 7.7537235550066748e+02
        self.fsv = 7.7537235550066748e+02
        self.cu = 6.1947309112548828e+02
        self.cv = 2.5718049049377441e+02
        self.stheta = 0#5.0510166700000003e-01
        self.mat = np.array([[self.fsu,self.stheta,self.cu,0],[0,self.fsv,self.cv,0],[0,0,0,self.fsu*self.baseline]])
        self.UL = np.arange(0,560,1)
        self.VL = np.arange(0,1280,1)
        self.meshx, self.meshy = np.meshgrid(self.VL, self.UL)
        self.xyz = np.zeros((560,1280,3))
        # self.VehicleFrameCoords = self.PicToWorld
    def lidarTosth2(self, lidarInV, disparity):
        xyz = self.oRr @ self.StereoToV_R @ (lidarInV-self.StereoToV_T.reshape(3,1)) # 3,286
        xyz1 = np.append(xyz, np.ones(lidarInV.shape[1]).reshape(1, lidarInV.shape[1]),axis=0) # 4,286
        ans = self.mat @ (xyz1/xyz[-1])
        print(np.max(ans))
        return 0,1
    def lidarTosth(self, lidarInV, disparity): # should have Z
        # 286by3 to 286by4 adding col of 1's
        # lidarInV = np.append(lidarInV, np.ones(lidarInV.shape[0]))
        # ZL = self.l_camera_intrinsicMat @ (self.self.StereoToV_R @ lidarInV + self.self.StereoToV_T)
        # ZR = self.r_camera_intrinsicMat @ (self.self.StereoToV_R @ lidarInV + self.self.StereoToV_T)
        # a = (lidarInV.T - self.StereoToV_T).T
        # b = self.StereoToV_R @ a
        # print(b.shape)
        # c = self.canonicalProj(b)
        # d = self.l_camera_intrinsicMat @ c
        # ZL =  self.l_camera_intrinsicMat @ self.canonicalProj(self.oRr @ (self.StereoToV_R @ (lidarInV - self.StereoToV_T)))
        # ZR =  self.r_camera_intrinsicMat @ self.canonicalProj(self.oRr @ (self.StereoToV_R @ (lidarInV - self.StereoToV_T)))
        # print("lidar1",lidarInV.shape)

        lidarInV = np.append(lidarInV, np.ones(lidarInV.shape[1]).reshape(1,lidarInV.shape[1]), axis=0)
        # print("lidar2",lidarInV.shape)
        # tmp1 = self.oRr@self.StereoToV_R
        tmp1 = self.StereoToV_R
        # print("tmp1",tmp1.shape)
        tmp2 = -tmp1 @ self.StereoToV_T.reshape(3,1)
        # print("tmp2",tmp2.shape)
        tmp3 = np.append(tmp1, tmp2, axis=1)
        tmp4 = np.zeros(4).reshape(1,4)
        tmp4 = np.append(tmp3,tmp4, axis=0)
        # print("tmp4",tmp4.shape)
        tmp4[-1,-1] = 1
        print(tmp4)

        xyz1o = tmp4@(lidarInV)
        print("xyz1o",xyz1o.shape) # 4, 286
        zo = xyz1o[2]

        ZL = self.mat @ (xyz1o/zo)
        ZR = ZL
        # ZL = (self.l_camera_intrinsicMat) @ (xyz1o/zo)
        # ZR = (self.r_camera_intrinsicMat) @ (xyz1o/zo)
        print(ZL.shape, np.max(ZL[:-1]), np.min(ZL[:-1]))
        return ZL, ZR
        # xyz1 = 

        # return ZL, ZR
    def canonicalProj(self,x): # 3 by 286
        e = np.array([0,0,1]).reshape(3,x.shape[1])
        return (1/(e@x))@x
    # def drawPointsOnImg()
    def getDisparity(self, path_l, path_r):
        image_l = cv2.imread(path_l, 0)
        image_r = cv2.imread(path_r, 0)
        image_l = cv2.cvtColor(image_l, cv2.COLOR_BAYER_BG2BGR)
        image_r = cv2.cvtColor(image_r, cv2.COLOR_BAYER_BG2BGR)
        image_l_gray = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
        image_r_gray = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)
        # You may need to fine-tune the variables `numDisparities` and `blockSize` based on the desired accuracy
        stereo = cv2.StereoBM_create(numDisparities=32, blockSize=9) 
        disparity = stereo.compute(image_l_gray, image_r_gray)
        # print("disparity",disparity.shape)
        return disparity
    def PicToVehicle(self, path_l, path_r):
        image_l = cv2.imread(path_l, 0)
        image_r = cv2.imread(path_r, 0)
        image_l = cv2.cvtColor(image_l, cv2.COLOR_BAYER_BG2BGR)
        image_r = cv2.cvtColor(image_r, cv2.COLOR_BAYER_BG2BGR)
        image_l_gray = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
        image_r_gray = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)
        # You may need to fine-tune the variables `numDisparities` and `blockSize` based on the desired accuracy
        stereo = cv2.StereoBM_create(numDisparities=32, blockSize=9) 
        disparity = stereo.compute(image_l_gray, image_r_gray)
        
        Z = self.fsu*self.baseline / disparity # 560,1280
        Z[Z> 50] = 0
        # print("ZZZZ", np.max(Z))
        # threshold Z
        # xyz = np.zeros((560,1280,3))
        # xyz1 = []
        
        ##### Optmize
        
        Y1 = Z * (self.meshx - self.cu)/self.fsu
        X1 = Z * (self.meshy - self.cv)/self.fsv
        self.xyz = np.zeros((560,1280,3))
        self.xyz[:,:,0] = X1
        self.xyz[:,:,1] = Y1
        self.xyz[:,:,2] = Z
        # print("hi")
        # print(xyz.shape)
        # xyz=xyz.T
        # xyz.reshape(560*1280,3)
        # xyz.reshape(3,(560*1280))
        self.xyz = self.xyz.transpose(2,0,1).reshape(3,-1).T
        # print(xyz.shape)

        #####


        # for i in range(1280):
        #     for j in range(560):
        #         X = Z[j,i] * (j-self.cv)/self.fsv
        #         Y = Z[j,i] * (i-self.cu)/self.fsu
        #         xyz[j,i] = np.array([X,Y,Z[j,i]])
        #         xyz1.append([X,Y,Z[j,i]])


        # print("mxboi1",np.max(xyz[:,:,0]))
        # print("mxboi2",np.max(X1))
        # print("mxboi3",np.max(Y1))
        # print("mxboi4",np.max(xyz[:,:,1]))



        # xyz = np.array(xyz1) # N by 3

        VehicleFrameCoords = self.StereoToV_R @ (self.xyz.T) + self.StereoToV_T.reshape(3,1)
        # print("maxvehilec", np.max(VehicleFrameCoords))
        return VehicleFrameCoords, image_l.reshape(-1,3)


        # VL = np.arange(0,560,1)
        # UL = np.arange(0,1280,1)
        # X = Z@UL * (-self.cu/self.fsu) #560
        # print("X"*20, np.max(X))
        # Y = VL@Z * (-self.cv/self.fsv) #1280
        # print("Y"*20, np.max(Y))
        # print(X.shape, Y.shape, Z.shape)
        # xyz = []
        # for i in range(1280):
        #     for j in range(560):
        #         xyz.append([X[j], Y[i], Z[j,i]])
        # xyz = np.array(xyz) # N by 3

        # VehicleFrameCoords = self.StereoToV_R @ (xyz.T) + self.StereoToV_T.reshape(3,1)
        # print("maxvehilec", np.max(VehicleFrameCoords))
        # return VehicleFrameCoords, image_l.reshape(-1,3)

class camera:
    def __init__(self) -> None:
        self.initrinsicMatrix = np.array([])
#distance = focal_length * baseline distance / disparity