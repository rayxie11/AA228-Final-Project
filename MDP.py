import numpy as np
from collections import deque
from transition import action, transition_model
from tqdm import tqdm

class MDP:
    def __init__(self, init_s, goal_s, environment) -> None:
        self.init_s = np.array(init_s)              # Initial state
        self.goal_s = np.array(goal_s)              # Goal state
        self.env = environment                      # Environment
        self.x = self.env.x+1                       # x dim
        self.y = self.env.y+1                       # y dim
        self.z = self.env.z+1                       # z dim
        self.pi = np.zeros(self.x*self.y*self.z)    # Initialize policy
        self.U = np.zeros(self.x*self.y*self.z)     # Initialize value function


    def reward(self, s, a):
        '''
        Reward function
        '''
        if np.all(s == self.goal_s):
            return 1000
        elif self.env.check_state_in_obstacle(s) != -1:
            return -1000
        r = -np.linalg.norm(self.goal_s-s)
        return r

    
    def value_iteration(self, gamma):
        '''
        Gauss-Seidel value iteration
        '''
        U = self.U
        pi = self.pi
        action_S = action
        
        for i in tqdm(range(5)):
            #visited = np.zeros((self.x, self.y, self.z), dtype=bool)
            #to_visit = deque()
            #to_visit.append(self.goal_s)
            #while to_visit:
            for j in tqdm(range(self.x*self.y*self.z)):
                s = np.unravel_index(self.x*self.y*self.z - 1 - j, (self.x, self.y, self.z))
                #s = to_visit.popleft()
                #visited[s] = True
                U_star = float('-inf')
                a_star = 0
                for idx, a in enumerate(action_S):
                    if self.env.check_state_in_bound(np.array(s)+np.array(a)):
                        #if not visited[tuple(s+a)]:
                        #    to_visit.append(s+a)
                        r = self.reward(s, a)
                        T = transition_model(s, a, self.env)
                        U_cur = r + gamma*T[0]@U[T[1]]
                        if U_cur > U_star:
                            U_star = U_cur
                            a_star = idx
                if np.all(s == self.goal_s):
                    U_star = 1000
                    a_star = 26
                U[np.ravel_multi_index(s, (self.x, self.y, self.z))] = U_star
                pi[np.ravel_multi_index(s, (self.x, self.y, self.z))] = a_star
        
        return U, pi
    

    
