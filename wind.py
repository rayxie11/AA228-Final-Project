import numpy as np
from box import Box

class Wind(Box):
    '''
    Wind is defined as a Box object with different directions with respective probabilities
    '''
    def __init__(self, dims, origin, prob, dir):
        super().__init__(dims, origin)
        self.prob = np.array(prob)     # Probability for each direction  
        self.dir = np.array(dir)       # Directions
        prob_expanded = np.tile(np.expand_dims(self.prob, axis=-1), (1,3))
        w = np.mean(prob_expanded*self.dir, axis=0)
        self.mean = w                  # Mean direction of wind

    def sample_wind(self):
        '''
        Sample the wind according to the probability assigned to each direction
        Return:
            wind_dir: wind direction
        '''
        idx_arr = np.linspace(0, len(self.dir)-1, num=len(self.dir))
        dir_idx = int(np.random.choice(idx_arr, 1, p=self.prob)[0])
        return self.dir[dir_idx]
        
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

        
