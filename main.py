import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from box import Box
from environment import Environment
from wind import Wind
from quadcopter import Quadcopter
from qlearning import QLearning

env_origin = [0,0,0]
env_dim = [20,20,20]

obs_origin = [3,3,3]
obs_dim = [4,4,4]
obs = [Box(obs_dim,obs_origin)]

w_origin = [10,10,10]
w_dim = [5,5,5]
prob = [0.5,0.3,0.1,0.1]
dir = [[1,1,1],[2,2,2],[3,3,3],[-3,-3,-3]]
w = [Wind(w_dim,w_origin,prob,dir)]
env = Environment(env_dim,env_origin,obs,w)

#'''
start = [1,1,1]
end = [17,17,17]
q = QLearning(start, end, env)
q.naive_qlearning(0.1,0.95)
print("finished q learning")
#q.trajectory = np.array(q.trajectory)
traj = q.generate_trajectory()
#print(traj.shape)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
env.plot_env(ax)
ax.scatter(1,1,1,c='r')
ax.scatter(15,15,15,c='g')
ax.scatter(traj[:,0],traj[:,1],traj[:,2],c='black')
#ax.scatter(q.trajectory[:,0],q.trajectory[:,1],q.trajectory[:,2],c='black')
ax.set_xlim([-1,30])
ax.set_ylim([-5,30])
ax.set_zlim([-5,30])
#ax.set_xticks([])
#ax.set_yticks([])
#ax.set_zticks([])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()