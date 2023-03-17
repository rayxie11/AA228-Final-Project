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
#dir = [[-3,-3,-3],[-2,-2,-2],[-1,-1,-1],[-5,-5,-5]]
w = [Wind(w_dim,w_origin,prob,dir)]

lrs = np.array([0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.14,0.16,0.18,0.2,0.25,0.3,0.4,0.45,0.5])
steps = []

for lr in lrs:
    env = Environment(env_dim,env_origin,obs,w)
    start = [1,1,1]
    end = [17,17,17]
    q = QLearning(start, end, env)
    q.naive_qlearning(lr,0.95)
    print("finished q learning with lr", lr)
    traj, stp = q.generate_trajectory()
    steps.append(stp)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    env.plot_env(ax)
    ax.scatter(1,1,1,c='r',label='inital state')
    ax.scatter(17,17,17,c='g',label='goal state')
    ax.scatter(traj[:,0],traj[:,1],traj[:,2],c='black',label='trajectory')
    ax.set_xlim([-5,25])
    ax.set_ylim([-5,25])
    ax.set_zlim([-5,25])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.savefig(str(lr)+'.png')

plt.plot(lrs,np.array(steps))
plt.xlabel('Learing Rate')
plt.ylabel('Number of Steps')
plt.title('Learning Rate vs. Steps taken to reach Goal State from generated policy')
plt.show()
