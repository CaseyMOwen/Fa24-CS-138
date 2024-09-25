# Casey Owen
# CS138
# Programming Assignment 1, 10-armed Bandit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from bandit import Bandit
from agent import Agent

def main():
    '''
    Creates format for storing the results, then runs the simulations and produces output figures
    '''
    
    n_runs_each = 2000
    sim_length = 10000
    
    # Build result dicitonary format
    results = {}
    for output_type in ["average_rewards", "optimal_action_pct"]:
        results[output_type] = {'ucb':{}, 'epsilon_greedy': {}}
        for as_method in results[output_type]:
            results[output_type][as_method] = {'sample_average': [], 'constant_alpha': []}
     
    # Only need to loop over first level of dictionary, since adding to both levels at each loop
    for as_method in results["average_rewards"]:
        for av_method in results["average_rewards"][as_method]:
            results["average_rewards"][as_method][av_method], results["optimal_action_pct"][as_method][av_method] = repeatedly_simulate(n_runs_each, sim_length, av_method, as_method)

    # Plot each result
    for output_type in results:
        for as_method in results[output_type]:
            df = pd.DataFrame(results[output_type][as_method])
            plot_results(df, output_type, as_method)

def repeatedly_simulate(n_runs:int, sim_length:int, av_method:str, as_method:str):
    '''
    Run a simulation n_runs times, and return the average results over all runs.
    '''
    n_arms = 10
    actions = ["Arm " + str(i) for i in range(n_arms)]
    # Initialize all q_stars as 0
    arm_q_stars = dict(zip(actions, [0]*n_arms))
    avg_rewards = np.zeros(sim_length)
    avg_optim_act = np.zeros(sim_length)
    for i in range(n_runs):
        agent = Agent(available_actions=actions)
        bandit = Bandit(arm_q_stars)
        rewards, optim_acts = run_simulation(agent, bandit, sim_length, av_method, as_method)
        avg_rewards += rewards/n_runs
        avg_optim_act += optim_acts/n_runs
        print(f'run {i}')
    return avg_rewards, avg_optim_act


def run_simulation(agent: Agent, bandit: Bandit, sim_length:int, av_method:str, as_method:str):
    '''
    Run a bandit problem simulation of a given length, action-value method, and action-selection method, and return the results
    '''
    rewards = np.zeros(sim_length)
    optim_acts = np.zeros(sim_length)
    for t in range(sim_length):
        action = agent.choose_action(as_method, epsilon=0.1, c=2, time=t+1)
        reward = bandit.choose_arm(action)
        rewards[t] = reward
        optim_acts[t] = int(action in bandit.best_actions)
        agent.learn(av_method, action, reward, alpha=0.1)
        bandit.random_walk(std=0.01)
    return rewards, optim_acts

def plot_results(df:pd.DataFrame, output_type:str, as_method:str):
    '''
    Create output plots
    '''
    fname = as_method + '_' + output_type + '.png'
    df.rename(columns={'sample_average': r'Sample Average ($\alpha=\frac{1}{n}$)', 'constant_alpha':r'Constant Alpha ($\alpha=.1$)'}, inplace=True)
    output_map = {"average_rewards": "Average Rewards", "optimal_action_pct": "Optimal Action %"}
    as_map = {'ucb': 'Upper-Confidence-Bound ' + output_map[output_type] + r' ($c=2$)', 'epsilon_greedy': '$\epsilon$-Greedy ' + output_map[output_type] + r' ($\epsilon=.1$)'}
    df.plot()
    if output_type == "optimal_action_pct":
        plt.ylim(0,1)        
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.xlabel('Steps')
    plt.ylabel(output_map[output_type])
    plt.title(as_map[as_method])
    plt.savefig(fname)

if __name__ == "__main__":
    main()