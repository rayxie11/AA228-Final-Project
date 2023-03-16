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
        #self.trajectory = []                       # Trajectory generated from Q table
    
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
        '''
        for obs in self.env.obstacles:
            x = np.linspace(obs.origin[0], obs.origin[0]+obs.x, num=obs.x+1).astype('int64')
            y = np.linspace(obs.origin[1], obs.origin[1]+obs.y, num=obs.y+1).astype('int64')
            z = np.linspace(obs.origin[2], obs.origin[2]+obs.z, num=obs.z+1).astype('int64')
            XX,YY,ZZ = np.meshgrid(x,y,z)
            all_state = np.stack([XX,YY,ZZ],axis=-1)
            tot = len(x)*len(y)*len(z)
            all_state = np.reshape(all_state, (tot,3))
            for state in all_state:
                self.Q_table[state[0],state[1],state[2]] = np.ones(self.quadcopter.a_count)*(-10000)
        '''
        
    
    def naive_reward(self, quadcopter):
        '''
        Naive reward function considering distance from goal and in wind region or not
        '''
        dist = np.linalg.norm(self.goal_s-quadcopter.s)
        wind_idx = self.quadcopter.in_wind(self.env)
        '''
        wind_reward = 1
        if wind_idx != -1:
            w = self.env.wind[wind_idx].mean
            w /= np.linalg.norm(w)
            diff = np.linalg.norm(w-self.goal_s/np.linalg.norm(self.goal_s))
            wind_reward = 1/np.exp(diff)*10
        reward = 1/np.exp(dist)*1000+wind_reward
        '''
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
        num_state_to_explore = self.env.x*self.env.y*self.env.z
        epoch = 0
        x = np.linspace(self.init_s[0], self.env.x, num=self.env.x-self.init_s[0]+1).astype('int64')
        y = np.linspace(self.init_s[1], self.env.y, num=self.env.y-self.init_s[1]+1).astype('int64')
        z = np.linspace(self.init_s[2], self.env.z, num=self.env.z-self.init_s[2]+1).astype('int64')
        XX,YY,ZZ = np.meshgrid(x,y,z)
        all_state = np.stack([XX,YY,ZZ],axis=-1)
        tot = len(x)*len(y)*len(z)
        all_state = np.reshape(all_state, (tot,3))
        print(all_state.shape)
        j = 0
        for state in all_state:
            print(j)
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
        '''
        while epoch < 50:
            print(epoch)
            random_x = np.random.randint(self.init_s[0], self.env.x+1)
            random_y = np.random.randint(self.init_s[1], self.env.y+1)
            random_z = np.random.randint(self.init_s[2], self.env.z+1)
            cur_quadcopter = Quadcopter([random_x,random_y,random_z])
            i = 0
            while i < num_state_to_explore:
                cur_state = cur_quadcopter.s.copy()
                eps = np.random.random(1)[0]
                if eps < 0.7:
                    cur_action = cur_quadcopter.next_state_with_random_exploration(self.env)
                else:
                    cur_action = cur_quadcopter.next_state(self.env)
                next_max_q = np.max(self.Q_table[cur_quadcopter.s[0],cur_quadcopter.s[1],cur_quadcopter.s[2]])
                Q_val = self.Q_table[cur_state[0],cur_state[1],cur_state[2],cur_action]+ \
                            lr*(self.naive_reward(cur_quadcopter)+gamma*next_max_q-self.Q_table[cur_state[0],cur_state[1],cur_state[2],cur_action])
                self.Q_table[cur_state[0],cur_state[1],cur_state[2],cur_action] = Q_val
                i += 1
            epoch += 1
        '''

    
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
        while np.linalg.norm(cur_state-np.array(self.goal_s)) > np.sqrt(3):
            print(cur_state)
            traj.append(cur_state)
            visited[cur_state[0],cur_state[1],cur_state[2]] = 1
            best_action_ls = np.argsort(self.Q_table[cur_state[0],cur_state[1],cur_state[2]])
            print(self.Q_table[cur_state[0],cur_state[1],cur_state[2]])
            print(best_action_ls)
            j = self.quadcopter.a_count-1
            pos_state = cur_state+action2move[best_action_ls[j]]
            while j > 0:
                if self.env.check_valid_state(pos_state):
                    if visited[pos_state[0],pos_state[1],pos_state[2]] == 0:
                        break
                j -= 1
                pos_state = cur_state+action2move[best_action_ls[j]]
            cur_state = pos_state
            #print(best_action_idx)
            #best_action = np.array(action2move[best_action_idx])
            #cur_state += best_action
            i += 1
            #break
        traj.append(cur_state)
        return np.array(traj)
        

    
