# Casey Owen
# CS138
# Programming Assignment 1, 10-armed Bandit

import numpy as np
from bandit import Bandit
from agent import Agent
import matplotlib.pyplot as plt

def main():
    n_arms = 10
    actions = ["Arm " + str(i) for i in range(n_arms)]
    # Initialize all q_stars as 0
    arm_q_stars = dict(zip(actions, [0]*n_arms))
    agent = Agent(available_actions=actions)
    bandit = Bandit(arm_q_stars)
    rewards = run_simulation(agent, bandit, 1000)
    plot_results(rewards, 'rewards_plot.png')

def repeatedly_simulate(n_runs, av_method, as_method):
    pass

def run_simulation(agent: Agent, bandit: Bandit, time_steps:int):
    rewards = np.zeros(time_steps)
    for t in range(time_steps):
        action = agent.choose_action('epsilon greedy', .1, t+1)
        reward = bandit.choose_arm(action)
        rewards[t] = reward
        agent.learn("sample average", action, reward)
        bandit.random_walk(std=0.01)
    return rewards

def plot_results(rewards, fname):
    fig, ax = plt.subplots()
    ax.plot(rewards)
    # ax.set_title('Likelihoods of Birds and Planes Given Velocity')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Average Reward')
    # ax.set_ylim(ymin=0)
    # ax.legend(['Planes', 'Birds'])
    fig.savefig(fname)


if __name__ == "__main__":
    main()