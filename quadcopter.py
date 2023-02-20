import numpy as np

# Quadcopter actions
actions = {0:(1,0,0), 1:(0,1,0), 2:(0,0,1),
           3:(-1,0,0), 4:(0,-1,0), 5:(0,0,-1),
           6:(1,1,1), 7:(1,-1,1), 8:(-1,1,1), 9:(-1,-1,1),
           10:(1,1,-1), 11:(1,-1,-1), 12:(-1,1,-1), 13:(-1,-1,-1)}
for a in actions.keys():
    actions[a] = np.array(actions[a])

class Quadcopter:
    def __init__(self, init_s):
        self.s = np.array(init_s)
    
    def generate_valid_actions(self, environment):
        '''
        Generate valid actionis that can be performed by the quadcopter
        Args:
            environment: environment quadcopter is in
        Return:
            valid_actions: valid actions the quadcopter can take in current state
        '''
        valid_actions = []
        for a in actions:
            potential_s = self.s+actions[a]
            if environment.check_state_in_bound(potential_s):
                if environment.check_state_in_obstacle(potential_s):
                    valid_actions.append(a)
        return valid_actions
