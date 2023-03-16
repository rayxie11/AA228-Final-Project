import numpy as np
import random

# Quadcopter actions
action = [(1,0,0),(-1,0,0),(0,1,0),(1,1,0),(-1,1,0),(0,1,1),
        (1,1,1),(-1,1,1),(0,0,1),(1,0,1),(-1,0,1),(0,-1,0),
        (1,-1,0),(-1,-1,0),(0,-1,-1),(1,-1,-1),(-1,-1,-1),
        (0,0,-1),(1,0,-1),(-1,0,-1),(0,1,-1),(1,1,-1),
        (-1,1,-1),(0,-1,1),(1,-1,1),(-1,-1,1),(0,0,0)]

# Action unit vector
direction = np.array(action)/(np.linalg.norm(action, axis = 1) + 1e-6).reshape((-1, 1))

def transition_model(s, a, env):
    '''
    State transition with uncertain wind
    '''
    in_wind = env.check_state_in_wind(s)
    p_list = []
    s_list = []
    if in_wind != -1:
        v = env.wind[in_wind].dir
        p = env.wind[in_wind].prob
        for i in range(len(v)):
            sp = np.array(s) + action[np.argmax(direction@(np.array(a)+v[i]))]
            if env.check_state_in_bound(sp):
                p_list.append(p[i])
                s_list.append(np.ravel_multi_index(sp, (env.x+1, env.y+1, env.z+1)))
    else:
        sp = np.array(s) + np.array(a)
        p_list.append(1.0)
        s_list.append(np.ravel_multi_index(sp, (env.x+1, env.y+1, env.z+1)))
    
    return [np.array(p_list), np.array(s_list)]

def simulate(start, end, pi, env):
    '''
    Simulate trajectory with learned policy
    '''
    x = env.x+1
    y = env.y+1
    z = env.z+1
    weight_l = []
    vector_l = []
    for region in env.wind:
        p = region.prob
        v = region.dir
        weight_l.append(p)
        vector_l.append(v)
    s = np.ravel_multi_index(np.array(start), (x, y, z))
    g = np.ravel_multi_index(np.array(end), (x, y, z))
    traj = []
    move = []
    wind_s = []
    wind_v = []
    while s != g:
        a = action[pi[s].astype('int')]
        s = np.unravel_index(s, (x, y, z))
        traj.append(s)
        move.append(a)
        in_wind = env.check_state_in_wind(s)
        if in_wind != -1:
            w = weight_l[in_wind]
            v = vector_l[in_wind]
            wind = random.choices(v, weights=w, k=1)
            wind_s.append(s)
            wind_v.append(wind[0])
            s = np.array(s) + action[np.argmax(direction@(np.array(a)+wind[0]))]
        else:
            s = np.array(s) + np.array(a)
        if env.check_state_in_obstacle(s):
            print('Crash!')
            break
        if not env.check_state_in_bound(s):
            print('Out of bound!')
            break
        s = np.ravel_multi_index(s, (x, y, z))
    return traj, move, wind_s, wind_v

def plot_arrow(s, v, ax, color):
    ax.quiver(s[:, 0], s[:, 1], s[:, 2], v[:, 0], v[:, 1], v[:, 2], length=0.8, color=color)
