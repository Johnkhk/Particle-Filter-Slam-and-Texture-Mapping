import numpy as np
import itertools
from pr2_utils import *
a = np.array([[1,2],[3,4]]) #2,2
# b = np.array([[5,6],[7,8],[9,10]]).T #2,3
# c = np.hstack((a,b))
# print(c.shape)
# print(c)

# a = np.array([1,2,3,4]) #2,2
# b = np.array([5,6]) #2,2

# c = np.sum(b.reshape(-1,1)*a)
# print(c)

# res = (a<3) *b*a
# # res = res[res!=0]
# # res = a(a<3) *b
# print(res)

#logodds.data[lidar_scan_indices[:, 1], lidar_scan_indices[:, 0]] += MAP_LOGODDS_OCCUPIED_DIFF - MAP_LOGODDS_FREE_DIFF

# a = np.array([[1,2,3],[4,5,6],[7,8,9]]) # 3 by 3
# b = np.array([[0,0],[1,1],[2,2]]) #2 by 3
# a=a.reshape(-1,3)
# print(a)


# # print(b[:,0])
# print(a[b[:,0], b[:,1]])

# a=3
# b=5
# a,b += 2
# print(a,b)

# bias = {}
# c = np.zeros(25)
# for j, (bx, by) in enumerate(
#         itertools.product(range(-2, 3), range(-2, 3))):
#     bias[j] = (bx, by)
# print(bias)

a = np.array([[1,2],[3,4]]) #2,2
# # b = np.array([1,1])
# # c = np.sum(a,b, axis=1)
# # c = a+b
# # indx = np.array([[0,0],[1,1]])
# indx = [0]

# # c = np.add.at(a, indx, 5)
# a[indx] += 1
# print(a)
b = np.array([50,50])
# np.concatenate((a,b))
# print(a)
# compute_stereo()

path_r = 'data/image_right.png'

image_r = cv2.imread(path_r, 0)
image_r = cv2.cvtColor(image_r, cv2.COLOR_BAYER_BG2BGR)
# image_r = cv2.cvtColor(image_r, cv2.COLOR_BAYER_BGR2RGBcv2 bgr)
plt.imshow(image_r)
plt.show(block = True)
plt.savefig("rgb_road1.png",dpi=1200)

path_r = 'data/image_right.png'

image_r = cv2.imread(path_r, 0)
image_r = cv2.cvtColor(image_r, cv2.COLOR_BAYER_BG2BGR)
# image_r = cv2.cvtColor(image_r, cv2.COLOR_BAYER_BGR2RGBcv2 bgr)
plt.imshow(image_r)
plt.savefig("rgb_road2.png",dpi=1200)

plt.show(block = True)

# UL = np.arange(0,560,1)
# VL = np.arange(0,1280,1)

# c = np.meshgrid(VL,UL)
# print(len(c))
# print(c[0].shape)
# print(c[1].shape)
# print(c[0])
# print(c[1])
