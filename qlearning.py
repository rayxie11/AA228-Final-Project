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
    
    def generate_Q_table(self):
        '''
        Generate initial Q table with all values set as 0
        '''
        x_dim = self.env.x+self.env.origin[0]+1
        y_dim = self.env.y+self.env.origin[1]+1
        z_dim = self.env.z+self.env.origin[2]+1
        self.Q_table = np.zeros((x_dim,y_dim,z_dim,self.quadcopter.a_count))
        # Set goal position without moving to have reward 10000
        self.Q_table[self.goal_s[0],self.goal_s[1],self.goal_s[2]] = np.ones(self.quadcopter.a_count)*10000
        
    
    def naive_reward(self, quadcopter):
        '''
        Naive reward function considering distance from goal and in wind region or not
        '''
        dist = np.linalg.norm(self.goal_s-quadcopter.s)
        reward = 1/np.exp(dist)*1000
        return reward
    
    def naive_qlearning(self, lr, gamma):
        '''
        Implement a naive Q Learning Scheme for finite horizon
        '''
        # First explore an initial trajectory that can reach the goal position
        cur_quadcopter = Quadcopter(self.init_s)
        j = 0
        while np.linalg.norm(cur_quadcopter.s-self.goal_s) > 0:
            cur_state = cur_quadcopter.s.copy()
            print(j, cur_state)
            eps = np.random.random(1)[0]
            if eps < 0.9:
                cur_action = cur_quadcopter.next_state_with_random_exploration(self.env)
            else:
                cur_action = cur_quadcopter.next_state(self.env)
            next_max_q = np.max(self.Q_table[cur_quadcopter.s[0],cur_quadcopter.s[1],cur_quadcopter.s[2]])
            Q_val = self.Q_table[cur_state[0],cur_state[1],cur_state[2],cur_action]+ \
                    lr*(self.naive_reward(cur_quadcopter)+gamma*next_max_q-self.Q_table[cur_state[0],cur_state[1],cur_state[2],cur_action])
            self.Q_table[cur_state[0],cur_state[1],cur_state[2],cur_action] = Q_val
            j += 1
        
        # To fill up the Q table, start from a random state and explore for a number of times
        x = np.linspace(self.init_s[0], self.env.x, num=self.env.x-self.init_s[0]+1).astype('int64')
        y = np.linspace(self.init_s[1], self.env.y, num=self.env.y-self.init_s[1]+1).astype('int64')
        z = np.linspace(self.init_s[2], self.env.z, num=self.env.z-self.init_s[2]+1).astype('int64')
        XX,YY,ZZ = np.meshgrid(x,y,z)
        all_state = np.stack([XX,YY,ZZ],axis=-1)
        tot = len(x)*len(y)*len(z)
        all_state = np.reshape(all_state, (tot,3))
        j = 0
        for state in all_state:
            if not self.env.check_valid_state(state):
                j += 1
                continue
            cur_quadcopter = Quadcopter(state)
            i = 0
            while i < 200:
                cur_state = cur_quadcopter.s.copy()
                eps = np.random.random(1)[0]
                if eps < 0.7:
                    cur_action = cur_quadcopter.next_state_with_random_exploration(self.env)
                else:
                    cur_action = cur_quadcopter.next_state(self.env)
                if cur_action == -1:
                    i += 1
                    continue
                next_max_q = np.max(self.Q_table[cur_quadcopter.s[0],cur_quadcopter.s[1],cur_quadcopter.s[2]])
                Q_val = self.Q_table[cur_state[0],cur_state[1],cur_state[2],cur_action]+ \
                            lr*(self.naive_reward(cur_quadcopter)+gamma*next_max_q-self.Q_table[cur_state[0],cur_state[1],cur_state[2],cur_action])
                self.Q_table[cur_state[0],cur_state[1],cur_state[2],cur_action] = Q_val
                i += 1
            j += 1

    
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
        while np.linalg.norm(cur_state-np.array(self.goal_s)) > np.sqrt(3):
            traj.append(cur_state)
            visited[cur_state[0],cur_state[1],cur_state[2]] = 1
            best_action_ls = np.argsort(self.Q_table[cur_state[0],cur_state[1],cur_state[2]])
            j = self.quadcopter.a_count-1
            pos_state = cur_state+action2move[best_action_ls[j]]
            while j > 0:
                if self.env.check_valid_state(pos_state):
                    if visited[pos_state[0],pos_state[1],pos_state[2]] == 0:
                        break
                j -= 1
                pos_state = cur_state+action2move[best_action_ls[j]]
            cur_state = pos_state
            i += 1
        traj.append(cur_state)
        return np.array(traj), i
        

    
