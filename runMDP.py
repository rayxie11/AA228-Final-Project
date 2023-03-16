import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from box import Box
from environment import Environment
from wind import Wind
from MDP import MDP
from transition import simulate, plot_arrow

def plot_result(traj, move, wind_s, wind_v, env, start, end):
    traj = np.vstack(traj)
    move = np.vstack(move)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    env.plot_env(ax)
    ax.scatter(start[0],start[1],start[2],c='r')
    ax.scatter(end[0],end[1],end[2],c='g')

    ax.scatter(traj[:,0],traj[:,1],traj[:,2],c='black')
    plot_arrow(traj, move, ax, 'black')
    if len(wind_s)>1:
        wind_s = np.vstack(wind_s)
        wind_v = np.vstack(wind_v)
        plot_arrow(wind_s, wind_v, ax, 'red')
    ax.set_xlim([-5,env.x+5])
    ax.set_ylim([-5,env.y+5])
    ax.set_zlim([-5,env.z+5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

start = [1,1,1]
end = [19,19,5]

env_origin = [0,0,0]
env_dim = [20,20,10]

obs_origin = [5,8,1]
obs_dim = [4,4,8]

w_origin = [4,3,3]
w_dim = [5,5,5]
prob = [0.9,0.1,0.0,0.0]
dir = [(1, 1, 0),
        (0.5, 0.5, 0),
        (0.5, 0, 0.5),
        (0.5, 0.5, -0.5)]
w = [Wind(w_dim,w_origin,prob,dir)]
obs = [Box(obs_dim,obs_origin)]
env = Environment(env_dim,env_origin,obs,w)

# Train with wind
n_iter = 20
mdp = MDP(start, end, env, n_iter)
U, pi, U_sum = mdp.value_iteration(0.95)
plt.plot(U_sum)
plt.show()

traj, move, wind_s, wind_v = simulate(start, end, pi, env)
plot_result(traj, move, wind_s, wind_v, env, start, end)

# Train without wind
env = Environment(env_dim,env_origin,obs,[])
mdp = MDP(start, end, env, n_iter)
U, pi, U_sum = mdp.value_iteration(0.95)
plt.plot(U_sum)
plt.show()

env = Environment(env_dim,env_origin,obs,w)
traj, move, wind_s, wind_v = simulate(start, end, pi, env)
plot_result(traj, move, wind_s, wind_v, env, start, end)
