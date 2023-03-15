import numpy as np
from box import Box

class Wind(Box):
    '''
    Wind is defined as a constant force field defined by a normal distribution of the given mean and
    standard deviation. The force of the wind is sampled every time when called. 
    '''
    def __init__(self, dims, origin, wind_mean, wind_std):
        super().__init__(dims, origin)
        self.mean = np.array(wind_mean)    # Wind vector mean
        self.cov = np.diag(wind_std)       # Wind vector covariance matrix

    def sample_wind(self):
        '''
        Sample wind force
        Return:
            sampled_wind: sampled wind force vector according a multivariate normal distribution
        '''
        sampled_wind = np.random.multivariate_normal(self.mean, self.cov)
        return sampled_wind
        
    def discrete_wind(self):
        '''
        Descrete wind distribution
        Return:
            discrete_wind: list of tuple, (probability, (velocity))
        '''
        discrete_wind = [(0.5, (-1, 1, 1)),
                        (0.3, (0, 1, 1)),
                        (0.1, (-1, -1, -1)),
                        (0.1, (-0.5, 0.5, 0.5))]
        
        return discrete_wind
    
    def plot_wind(self, ax):
        '''
        Plot wind vectors in 3D space
        Args:
            ax: matplotlib.pyplot axes
        '''
        XX, YY, ZZ = np.meshgrid(np.arange(self.origin[0],self.origin[0]+self.x,2),
                                 np.arange(self.origin[1],self.origin[1]+self.y,2),
                                 np.arange(self.origin[2],self.origin[2]+self.z,2))
        w = self.mean/np.linalg.norm(self.mean)
        U = XX+w[0]
        V = YY+w[1]
        W = ZZ+w[2]
        ax.quiver(XX, YY, ZZ, U, V, W, length=0.05, color='red')

        
