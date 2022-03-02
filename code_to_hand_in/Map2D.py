import numpy as np
class Map2D:
    def __init__(self, xmin, xmax, ymin, ymax, res):
        self.MAP = {}
        self.MAP['res']   = res #meters
        self.MAP['xmin']  = xmin  #meters
        self.MAP['xmax']  =  xmax
        self.MAP['ymin']  = ymin
        self.MAP['ymax']  =  ymax 
        self.MAP['sizex']  = int(np.ceil((self.MAP['xmax'] - self.MAP['xmin']) / self.MAP['res'] + 1)) #cells
        self.MAP['sizey']  = int(np.ceil((self.MAP['ymax'] - self.MAP['ymin']) / self.MAP['res'] + 1))
        self.MAP['map'] = np.zeros((self.MAP['sizex'],self.MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8 
        self.MAP['mapLogOdds'] = np.zeros((self.MAP['sizex'],self.MAP['sizey'])) #DATA TYPE: char or int8
        self.MAP['mapPMF'] = np.zeros((self.MAP['sizex'],self.MAP['sizey'])) #DATA TYPE: char or int8
        self.MAP['texture'] = np.zeros((self.MAP['sizex'],self.MAP['sizey'],3)) #DATA TYPE: char or int8

    def coordToMap(self, coordinates):
        coordinates = np.array(coordinates)
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(1, -1)
        return np.hstack([
            np.ceil((coordinates[:, 0] - self.MAP['xmin']) / self.MAP['res']).reshape(
                -1, 1),
            np.ceil((coordinates[:, 1] - self.MAP['ymin']) / self.MAP['res']).reshape(
                -1, 1),
        ]).astype(np.int32)
    def in_map(self, coordinates):
        # coordinates = coordinates.T
        tmp = np.logical_and(
            np.logical_and(self.MAP['xmin'] <= coordinates[:, 0],
                           coordinates[:, 0] <= self.MAP['xmax']),
            np.logical_and(self.MAP['ymin'] <= coordinates[:, 1],
                           coordinates[:, 1] <= self.MAP['ymax']))
        return tmp