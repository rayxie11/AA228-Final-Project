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
            valid_action_mask: a mask for valid actions the quadcopter can take in current state
        '''
        # 1 if action can be performed, 0 otherwise
        valid_action_mask = np.zeros(len(action2move))
        for a in action2move.keys():
            potential_s = self.s+action2move[a]
            if environment.check_state_in_bound(potential_s):
                if environment.check_state_in_obstacle(potential_s) == -1:
                    valid_action_mask[a] = 1
        return valid_action_mask
    
    def in_wind(self, environment):
        '''
        See if the quadcopter is in wind region at current state
        Args:
            environment: environment quadcopter is in
        Return:
            i: index of wind region quadcopter is in (-1 if not in any)
        '''
        return environment.check_state_in_wind(self.s)
    
    def transition_probability(self, environment):
        '''
        Get the probability of transitioning to the next state without any control input
        The smaller different between wind direction and move direction, the higher probability
        the quadcopter would take that action
        Args:
            environment: environment quadcopter is in
        Return:
        '''
        probability = []
        valid_action_mask = self.generate_valid_actions(environment)
        wind_idx = self.in_wind(environment)
        if wind_idx == -1:
            probability = valid_action_mask/np.sum(valid_action_mask)
        else:
            w = environment.wind[wind_idx].sample_wind()
            w /= np.linalg.norm(w)
            for i in range(len(valid_action_mask)):
                if valid_action_mask[i] == 0:
                    probability.append(0)
                else:
                    a = action2move[i]
                    # Use np.exp to avoid divide by 0 exception
                    if np.linalg.norm(a) == 0:
                        diff = 1/np.exp(np.linalg.norm(w))
                    else:
                        diff = 1/np.exp(np.linalg.norm(w-a/np.linalg.norm(a)))
                    #print(np.linalg.norm(w-a/np.linalg.norm(a)))
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
        Calculate the next state with the transition probability
        Args:
            action: given action
            environment: environment quadcopter is in
        Return:
            action: index of actual action taken
        '''
        T = self.transition_probability(environment)
        print(T)
        action = np.random.choice(actions,1,p=T)[0]
        self.s = self.s+action2move[action]
        return action

    def next_state1(self, action, environment):
        '''
        Calculate the next state given action with the transition probability
        Args:
            action: given action
            environment: environment quadcopter is in
        Return:
            action: index of actual action taken
        '''
        p = np.random.random(1)[0]
        # If p <= 0.6, go with given action
        if p <= 0.6 and action != -1:
            self.s = self.s+action2move[action]
        # Otherwise, sample action from transition probability
        else:
            T = self.transition_probability(environment)
            print(T)
            action = np.random.choice(actions,1,p=T)[0]
            self.s = self.s+action2move[action]
        return action


