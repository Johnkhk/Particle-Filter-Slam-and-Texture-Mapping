from unicodedata import name
import numpy as np
from pr2_utils import *
import math
from numpy import deg2rad
from Map2D import *
import itertools
# Todo min & max lambda value
# take max out of 9
class slam:
    def __init__(self, numParticles, map_xlim, map_ylim, map_res, Nthresh):
        self.numParticles = numParticles
        self.map_xlim = map_xlim
        self.map_ylim = map_ylim
        self.map_res = map_res
        self.particles = np.zeros((numParticles, 3))
        self.weights = np.ones((numParticles)) / numParticles
        self.mapLogOdds = Map2D(*map_xlim, *map_ylim, map_res)
        self.bias = self.biasInit()
        self.Nthresh = Nthresh
    @property
    def OccupancyMap(self):
        return (self.mapLogOdds['mapLogOdds'] > 0).astype(np.int32)
    @property
    def mapPDF(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.mapLogOdds['mapLogOdds']))
    def biasInit(self):
        bias={}
        for j, (bx, by) in enumerate(itertools.product(range(-4, 5), range(-4, 5))):
            bias[j] = (bx, by)
        return bias
    def coordToMap(self, coordinates):
        return self.mapLogOdds.coordToMap(coordinates)
    
    def predict(self, data):
        pass

# class data:
#     def __init__(self):
#         self.theta=0
#         self.xcar, self.ycar =0, 0



if __name__ == '__main__':
    slam = slam(5, (-300, 1400), (-1100, 800), 1)