from kinematic_quadcopter import KinematicQuadcopter
import numpy as np
from box import Box
from environment import Environment
from wind import Wind

env_origin = [0,0,0]
env_dim = [20,20,20]

obs_origin = [5,3,3]
obs_dim = [4,4,4]
obs = [Box(obs_dim,obs_origin)]

w_origin = [12,12,12]
w_dim = [5,5,5]
mean = [1,1,1]
std = [0.1,0.1,0.1]
w = [Wind(w_dim,w_origin,mean,std)]

env = Environment(env_dim,env_origin,obs,w)

init_quadcopter = KinematicQuadcopter([0,0,0,0,1,0], [0,0,0,0,0,0], 1, 2, 0, 5)
x = init_quadcopter.sample(env)
print(len(x))