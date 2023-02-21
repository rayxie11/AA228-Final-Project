import numpy as np
from box import Box

class Wind(Box):
    def __init__(self, dims, origin, wind_vec):
        super().__init__(dims, origin)
        self.wind_vec = wind_vec
    
    def plot_wind(self, ax):
        '''
        Plot wind vectors in 3D space
        Args:
            ax: matplotlib.pyplot axes
        '''
        XX, YY, ZZ = np.meshgrid(np.arange(self.origin[0],self.origin[0]+self.x,2),
                                 np.arange(self.origin[1],self.origin[1]+self.y,2),
                                 np.arange(self.origin[2],self.origin[2]+self.z,2))
        w = self.wind_vec/np.linalg.norm(self.wind_vec)
        U = XX+w[0]
        V = YY+w[1]
        W = ZZ+w[2]
        ax.quiver(XX, YY, ZZ, U, V, W, length=0.05, color='red')

        
