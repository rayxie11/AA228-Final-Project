import numpy as np
from box import Box

class Environment(Box):
    def __init__(self, dims, origin, obstacles, wind):
        super().__init__(dims, origin)
        self.obstacles = obstacles
        self.wind = wind
    
    def check_state_in_bound(self, state):
        '''
        Check whether the given state is inside the environment
        Args:
            state: current state of quadcopter
        Return:
            True/False: inside the environment or not
        '''
        return self.check_point_inside(state)
    
    def check_state_in_obstacle(self, state):
        '''
        Check whether the given state is inside any obstacle
        Args:
            state: current state of quadcopter
        Return:
            i: index of obstacle quadcopter is in (-1 if not in any)
        '''
        for i in range(len(self.obstacles)):
            if self.obstacles[i].check_point_inside(state):
                return i
        return -1
    
    def check_state_in_wind(self, state):
        '''
        Check whether the given state is inside wind region
        Args:
            state: current state of quadcopter
        Return:
            i: index of wind region quadcopter is in (-1 if not in any)
        '''
        for i in range(len(self.wind)):
            if self.wind[i].check_point_inside(state):
                return i
        return -1
    
    def plot_env(self, ax):
        '''
        Plot the entire environment
        Args:
            ax: matplotlib.pyplot axes
        '''
        self.plot(ax, 'cyan', 0, 0.2)
        for obs in self.obstacles:
            obs.plot(ax, 'black', 0, 1)
        for w in self.wind:
            w.plot(ax, 'red', 0, 0.5)
            w.plot_wind(ax)
    
