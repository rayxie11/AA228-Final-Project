from kinematic_quadcopter import KinematicQuadcopter
import numpy as np
from wind import Wind

pos = np.zeros(6)
vel = np.zeros(6)
mass = 2
dim = 1
t_max = 10
t_min = 10
x = KinematicQuadcopter(pos,vel,mass,dim,t_max,t_min)
#x.update(0.5)
print(x.pos)
print(x.ori)

w = Wind([0,0,0],[3,3,3],[10,10,10],[1,1,1])
w1 = w.sample_wind()
print(w1)