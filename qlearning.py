import numpy as np
from quadcopter import Quadcopter
from quadcopter import action2move

class QLearning:
    def __init__(self, init_s, goal_s, environment) -> None:
        self.init_s = init_s                       # Initial state
        self.goal_s = goal_s                       # Goal state
        self.env = environment                     # Environment
        self.quadcopter = Quadcopter(self.init_s)  # Initialize quadcopter
        self.generate_Q_table()
        self.trajectory = []                       # Trajectory generated from Q table
    
    def generate_Q_table(self):
        '''
        Generate initial Q table with all values set as 0
        '''
        x_dim = self.env.x+self.env.origin[0]+1
        y_dim = self.env.y+self.env.origin[1]+1
        z_dim = self.env.z+self.env.origin[2]+1
        self.Q_table = np.zeros((x_dim,y_dim,z_dim,self.quadcopter.a_count))
        # Set goal position without moving to have reward 1000
        self.Q_table[self.goal_s[0],self.goal_s[1],self.goal_s[2]] = np.ones(self.quadcopter.a_count)*1000
    
    def naive_reward(self):
        '''
        Naive reward function considering distance from goal and in wind region or not
        '''
        dist = np.linalg.norm(self.goal_s-self.quadcopter.s)
        wind_idx = self.quadcopter.in_wind(self.env)
        wind_reward = 1
        if wind_idx != -1:
            w = self.env.wind[wind_idx].mean
            w /= np.linalg.norm(w)
            diff = np.linalg.norm(w-self.goal_s/np.linalg.norm(self.goal_s))
            wind_reward = 1/np.exp(diff)*10
        reward = 1/np.exp(dist)*100+wind_reward
        return reward
    
    def naive_qlearning(self, lr, gamma):
        '''
        Implement a naive Q Learning Scheme for finite horizon
        '''
        #while self.quadcopter.s != self.goal_s:
        i = 0
        while np.linalg.norm(self.quadcopter.s-self.goal_s) > 0:
            print(i, self.quadcopter.s)
            
            cur_state = self.quadcopter.s.copy()
            self.trajectory.append(cur_state)
            eps = np.random.random(1)[0]
            if eps < 0.7:
                cur_action = self.quadcopter.next_state_with_random_exploration(self.env)
            else:
                cur_action = self.quadcopter.next_state(self.env)
            '''
            if np.linalg.norm(self.Q_table[cur_state[0],cur_state[1],cur_state[2]]) == 0:
                cur_action = self.quadcopter.next_state_with_random_exploration(self.env, 0.4)
            else:
                cur_action = np.argmax(self.Q_table[cur_state[0],cur_state[1],cur_state[2]])
                self.quadcopter.naive_next_state(cur_action, self.env)
            print(cur_action)
            '''
            next_max_q = np.max(self.Q_table[self.quadcopter.s[0],self.quadcopter.s[1],self.quadcopter.s[2]])
            self.Q_table[cur_state[0],cur_state[1],cur_state[2],cur_action] = self.Q_table[cur_state[0],cur_state[1],cur_state[2],cur_action]+ \
                            lr*(self.naive_reward()+gamma*next_max_q-self.Q_table[cur_state[0],cur_state[1],cur_state[2],cur_action])
            i += 1
    
    def generate_trajectory(self):
        '''
        Generate trajectory according to Q table
        Return:
            traj: trajectory of quadcopter
        '''
        x_dim = self.env.x+self.env.origin[0]+1
        y_dim = self.env.y+self.env.origin[1]+1
        z_dim = self.env.z+self.env.origin[2]+1
        visited = np.zeros((x_dim,y_dim,z_dim))
        traj = []
        cur_state = np.array(self.init_s)
        i = 0
        #while np.linalg.norm(cur_state-np.array(self.goal_s)) != 0 and i < 100:
        while np.linalg.norm(cur_state-np.array(self.goal_s)) != 0:
            print(cur_state)
            traj.append(cur_state)
            visited[cur_state[0],cur_state[1],cur_state[2]] = 1
            best_action_ls = np.argsort(self.Q_table[cur_state[0],cur_state[1],cur_state[2]])
            j = 0
            pos_state = cur_state+action2move[best_action_ls[j]]
            while j < self.quadcopter.a_count-1:
                if self.env.check_valid_state(pos_state):
                    if visited[pos_state[0],pos_state[1],pos_state[2]] == 0:
                        break
                j += 1
                pos_state = cur_state+action2move[best_action_ls[j]]
            cur_state = pos_state
            #print(best_action_idx)
            #best_action = np.array(action2move[best_action_idx])
            #cur_state += best_action
            i += 1
            #break
        traj.append(cur_state)
        return np.array(traj)
        

    
