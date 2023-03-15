import numpy as np

# Quadcopter actions
action2move = {0:(1,0,0), 1:(-1,0,0), 2:(0,1,0), 3:(1,1,0), 4:(-1,1,0), 5:(0,1,1),
           6:(1,1,1), 7:(-1,1,1), 8:(0,0,1), 9:(1,0,1), 10:(-1,0,1), 11:(0,-1,0),
           12:(1,-1,0), 13:(-1,-1,0), 14:(0,-1,-1), 15:(1,-1,-1), 16:(-1,-1,-1),
           17:(0,0,-1), 18:(1,0,-1), 19:(-1,0,-1), 20:(0,1,-1), 21:(1,1,-1),
           22:(-1,1,-1), 23:(0,-1,1), 24:(1,-1,1), 25:(-1,-1,1)}
actions = []
for a in action2move.keys():
    action2move[a] = np.array(action2move[a])
    actions.append(a)

class Quadcopter:
    def __init__(self, s):
        self.s = np.array(s)
        self.a_count = len(actions)
    
    def generate_valid_actions(self, environment):
        '''
        Generate valid actions that can be performed by the quadcopter in current state
        Args:
            environment: environment quadcopter is in
        Return:
            valid_actions: a list of indices of valid actions the quadcopter can take
        '''
        valid_actions = []
        for a in action2move.keys():
            potential_s = self.s+action2move[a]
            if environment.check_valid_state(potential_s):
                valid_actions.append(a)
        return valid_actions
    
    def in_wind(self, environment):
        '''
        See if the quadcopter is in wind region at current state
        Args:
            environment: environment quadcopter is in
        Return:
            i: index of wind region quadcopter is in (-1 if not in any)
        '''
        return environment.check_state_in_wind(self.s)
    
    def transition_probability(self, environment, valid_actions):
        '''
        Get the probability of transitioning to the next state. The smaller difference between 
        wind direction and move direction, the higher probability the quadcopter would take 
        that action
        Args:
            environment: environment quadcopter is in
            valid_actions: valid actions the quadcopter can take in the current state
        Return:
        '''
        probability = []
        wind_idx = self.in_wind(environment)
        if wind_idx == -1:
            probability = np.ones(len(valid_actions))/len(valid_actions)
        else:
            wind_dir = environment.wind[wind_idx].sample_wind()
            wind_dir = wind_dir/np.linalg.norm(wind_dir)
            for action_idx in valid_actions:
                a = action2move[action_idx]
                # Use np.exp to avoid divide by 0 exception
                if np.linalg.norm(a) == 0:
                    diff = 1/np.exp(np.linalg.norm(wind_dir))
                else:
                    diff = 1/np.exp(np.linalg.norm(wind_dir-a/np.linalg.norm(a)))
                probability.append(diff)
            probability = probability/np.sum(probability)
        return np.array(probability)
    
    def naive_next_state(self, action, environment):
        '''
        Update quadcopter state given action
        '''
        potential_s = self.s+action2move[action]
        if environment.check_state_in_bound(potential_s):
                if environment.check_state_in_obstacle(potential_s) == -1:
                    self.s += action2move[action]

    def next_state(self, environment):
        '''
        Get the next state with the transition probability
        Args:
            action: given action
            environment: environment quadcopter is in
        Return:
            action: index of actual action taken
        '''
        valid_actions = self.generate_valid_actions(environment)
        T = self.transition_probability(environment, valid_actions)
        #print(T)
        action = np.random.choice(valid_actions,1,p=T)[0]
        self.s = self.s+action2move[action]
        return action

    def next_state_with_random_exploration(self, environment):
        '''
        Explore next state randomly
        Args:
            environment: environment quadcopter is in
            e: probability threshold
        Return:
            action: index of actual action taken
        '''
        valid_actions = self.generate_valid_actions(environment)
        action = np.random.choice(valid_actions,1)[0]
        self.s = self.s+action2move[action]
        return action


