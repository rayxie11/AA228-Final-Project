import numpy as np
from box import Box

class Wind(Box):
    def __init__(self, dims, origin, wind_vec):
        super().__init__(dims, origin)
        self.wind_vec = wind_vec
        
