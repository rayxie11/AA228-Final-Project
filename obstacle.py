import numpy as np
from box import Box

class Environment(Box):
    def __init__(self, dims, origin, obstacles, wind):
        super().__init__(dims, origin)
        self.obstacles = obstacles
        self.wind = wind


class Wind(Box):
    def __init__(self, dims, origin, wind_vec):
        super().__init__(dims, origin)
        self.wind_vec = wind_vec
        
