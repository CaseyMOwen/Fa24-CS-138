# Casey Owen
# CS138
# Programming Assignment 1, 10-armed Bandit

from arm import Arm
import numpy as np

class Bandit():
    '''
    Implementation of a simple multi-armed bandit where each arm has unit variance.
    '''
    def __init__(self, arm_q_stars: dict) -> None:
        # arm_q_stars is a dictionary where the key is the arm name, and the value is the q star of that arm
        self.arms = {arm_name: Arm(arm_q_stars[arm_name], 1) for arm_name in arm_q_stars}
        pass

    def choose_arm(self, arm_name) -> float:
        '''
        Gets the reward from a chosen arm
        '''
        return self.arms[arm_name].sample_reward()
    
    def random_walk(self, std):
        '''
        Randomly walk all arms in the bandit
        '''
        for arm_name in self.arms: self.arms[arm_name].random_walk(std)

    @property
    def best_actions(self):
        '''
        The list of actions that are tied for the best intrinsic value (mean reward)
        '''
        arm_names = list(self.arms.keys())
        arm_vals = [self.arms[arm_name].q_star_mean for arm_name in arm_names]
        best_actions = np.argwhere(arm_vals == np.max(arm_vals)).flatten().tolist()
        return [arm_names[i] for i in best_actions]
