# Casey Owen
# CS138
# Programming Assignment 1, 10-armed Bandit

from arm import Arm

class Bandit():
    '''
    Implementation of a simple multi-armed bandit where each arm has unit variance.
    '''
    def __init__(self, arm_q_stars: dict) -> None:
        # arm_q_stars is a dictionary where the key is the arm name, and the value is the q star of that arm
        self.arms = {arm_name: Arm(arm_q_stars[arm_name], 1) for arm_name in arm_q_stars}

    def choose_arm(self, arm_name:int) -> float:
        '''
        Gets the reward from a chosen arm
        '''
        return self.arms[arm_name].sample_reward()
    
    def random_walk(self, std):
        '''
        Randomly walk all arms in the bandit
        '''
        for arm_name in self.arms: self.arms[arm_name].random_walk(std)