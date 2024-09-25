# Casey Owen
# CS138
# Programming Assignment 1, 10-armed Bandit

from arm import Arm
import numpy as np
import random

class Agent():
    '''
    Implementation of an agent in a simple multi-armed bandit scenario where the agent's decision does not impact the state.
    '''
    def __init__(self, available_actions: list) -> None:
        # However the available actions are passed in (strings, indexes, etc) we want to internally represent them as a list of integers, so save a mapping
        self.idx_to_action = {}
        self.action_to_idx = {}
        for i, action in enumerate(available_actions):
            # Key is list index, value is external representation of action
            self.idx_to_action[i] = action
            self.action_to_idx[action] = i
        # Agent has no initial knowledge of action values
        self.act_vals = [0]*len(available_actions)
        self.act_times_selected = [0]*len(available_actions)


    def choose_best_action(self):
        '''
        Uniformly randomly chooses an action among those tied for the best action value
        '''
        # All actions tied for best value
        best_actions = np.argwhere(self.act_vals == np.max(self.act_vals)).flatten().tolist()
        return self.idx_to_action[random.choice(best_actions)]
    
    def choose_random_action(self):
        '''
        Uniformly randomly chooses an action among all possible actions
        '''
        return random.choice(list(self.action_to_idx.keys()))
    
    def choose_epsilon_greedy(self, epsilon):
        '''
        Choose an action according to an epsilon-greedy strategy
        '''
        if np.random.uniform() < epsilon:
            return self.choose_random_action()
        else:
            return self.choose_best_action()
        
    def choose_UCB(self, c, time):
        '''
        Choose an action according to an Upper-Confidence-Bound strategy
        '''
        # If anything has been chosen 0 times, that is immediately the best action
        best_actions = np.flatnonzero(np.array(self.act_times_selected) == 0).tolist()
        # If none are 0, calculate ucb values according to formula
        if not best_actions:
            ucb_vals = np.array(self.act_vals) + c*np.sqrt(np.log(time)/np.array(self.act_times_selected))
            best_actions = np.argwhere(ucb_vals == np.max(ucb_vals)).flatten().tolist()
        return self.idx_to_action[random.choice(best_actions)]

    def choose_action(self, as_method:str, epsilon:float=None, c:float=None, time:int=None):
        '''
        Choose an action according to a a given action-selection method, with provided necessary parameters
        '''
        match as_method:
            case "epsilon_greedy":
                if epsilon == None:
                    raise ValueError("If using an action selection method of epsilon greedy, you must specify the parameter epsilon.")
                # if time==1:
                #     print()
                return self.choose_epsilon_greedy(epsilon)
            case "ucb":
                if c == None:
                    raise ValueError("If using an action selection method of UCB, you must specify the parameter c.")
                return self.choose_UCB(c, time)

    
    def learn(self, av_method:str, action, reward: float, alpha:float=None) -> None:
        '''
        Update agents knowledge of action values by providing it with the reward provided from a given action, and using the appropriate action-value method
        '''
        act_idx = self.action_to_idx[action]
        self.act_times_selected[act_idx] += 1
        match av_method:
            case "sample_average":
                # Alpha is 1/n
                alpha = (1/self.act_times_selected[act_idx])
                self.learn_incremental(act_idx, reward, alpha)
            case "constant_alpha":
                # User specified alpha
                if alpha == None:
                    raise ValueError("If using an action value method of constant alpha, you must specify the parameter alpha.")
                self.learn_incremental(act_idx, reward, alpha)

    def learn_incremental(self, act_idx: int, reward: float, alpha: float) -> None:
        '''
        Learn using the sample-average action value method, incrementally computed
        '''
        # Using incremental update formula
        self.act_vals[act_idx] += alpha*(reward - self.act_vals[act_idx])

