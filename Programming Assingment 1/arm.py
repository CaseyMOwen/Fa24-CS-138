# Casey Owen
# CS138
# Programming Assignment 1, 10-armed Bandit

import numpy as np

class Arm():
    '''
    Implementation of an arm of a simple multi-armed bandit. The reward has a user-specified mean and variance.
    '''
    def __init__(self, q_star_mean: float, q_star_var: float) -> None:
        self.q_star_mean = q_star_mean
        self.q_star_var = q_star_var

    def sample_reward(self) -> float:
        '''
        Sample the q_star distribution of the arm
        '''
        return np.random.normal(self.q_star_mean, np.sqrt(self.q_star_var))
    
    def random_walk(self, std) -> None:
        '''
        Changes the q_star distribution in a random walk fashion with given standard deviation 
        '''
        self.q_star_mean += np.random.normal(0, std)

