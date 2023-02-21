import numpy as np

# Quadcopter actions
action2move = {0:(1,0,0), 1:(0,1,0), 2:(0,0,1),
           3:(-1,0,0), 4:(0,-1,0), 5:(0,0,-1),
           6:(1,1,1), 7:(1,-1,1), 8:(-1,1,1), 9:(-1,-1,1),
           10:(1,1,-1), 11:(1,-1,-1), 12:(-1,1,-1), 13:(-1,-1,-1)}
actions = []
for a in action2move.keys():
    action2move[a] = np.array(action2move[a])
    actions.append(a)

class Quadcopter:
    def __init__(self, init_s):
        self.s = np.array(init_s)
    
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
    
    def transition_probability(self, enviroment):
        '''
        Get the probability of transitioning to the next state without any control input
        The smaller different between wind direction and move direction, the higher probability
        the quadcopter would take that action
        Args:
            environment: environment quadcopter is in
        Return:
        '''
        probability = []
        valid_action_mask = self.generate_valid_actions(enviroment)
        wind_idx = self.in_wind(enviroment)
        if wind_idx == -1:
            probability = valid_action_mask/np.sum(valid_action_mask)
        else:
            w = enviroment.wind[wind_idx].wind_vec
            w /= np.linalg.norm(w)
            for i in range(len(valid_action_mask)):
                if valid_action_mask[i] == 0:
                    probability.append(0)
                else:
                    a = action2move[i]
                    diff = 1/np.linalg.norm(w-a/np.linalg.norm(a))
                    print(diff)
                    probability.append(diff)
            probability = probability/np.sum(probability)
        return probability


