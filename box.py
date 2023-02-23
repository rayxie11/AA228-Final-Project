import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Box:
    def __init__(self, dims, origin):
        # Set dimensions of environment
        self.x = dims[0]
        self.y = dims[1]
        self.z = dims[2]
        self.origin = np.array(origin)
        self.generate_boundary_points()

    def check_point_inside(self, point):
        '''
        Check if given point is in the box
        Args:
            point: coordinates in 3D
        Return: 
            True/False: inside the box or not
        '''
        if point[0]<self.origin[0] or point[0]>self.x+self.origin[0] or \
            point[1]<self.origin[1] or point[1]>self.y+self.origin[1] or \
                point[2]<self.origin[2] or point[2]>self.z+self.origin[2]:
                    return False
        return True
    
    def generate_boundary_points(self):
        '''
        Construct boundary points for plotting
        '''
        self.p1 = np.array([0,0,0])+self.origin
        self.p2 = np.array([self.x,0,0])+self.origin
        self.p3 = np.array([self.x,self.y,0])+self.origin
        self.p4 = np.array([0,self.y,0])+self.origin
        self.p5 = np.array([0,0,self.z])+self.origin
        self.p6 = np.array([self.x,0,self.z])+self.origin
        self.p7 = np.array([self.x,self.y,self.z])+self.origin
        self.p8 = np.array([0,self.y,self.z])+self.origin
    
    def plot_box(self, ax, color, linewidth, a):
        '''
        Plot box in 3D space
        Args:
            ax: matplotlib.pyplot axes
            color: color of polygon
            linewidth: width of boundary line
            a: alpha
        '''
        verts = [[self.p1,self.p2,self.p3,self.p4],
                 [self.p5,self.p6,self.p7,self.p8],
                 [self.p2,self.p3,self.p7,self.p6],
                 [self.p3,self.p4,self.p8,self.p7],
                 [self.p1,self.p2,self.p6,self.p5],
                 [self.p1,self.p4,self.p8,self.p5],]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=color, linewidths=linewidth, edgecolors='r', alpha=a))





        


