# Author
# Rahul Kumar (Northeastern University)

import matplotlib.pyplot as plt
import gym
from env_v5 import register_env, tetherWorldEnv
from algorithm_v5 import q_learning, test_algorithm, load_checkpoint, on_policy_mc_control_epsilon_soft, SARSA
import numpy as np
from tqdm import trange
from scheduler import ExponentialSchedule
from plot import plot_learning_curve, plot_return
from math import pi

def main():
    register_env()
    tether_env = gym.make("tetherWorld-v0",req_winding_angle=2*pi)

    # parameters
    num_trials = 3
    # num_episodes = 10000
    num_episodes = 10000


    # hyperparameters
    gamma = 0.99
    step_size = 0.3
    # epsilon = 0.2
    epsilon = ExponentialSchedule(0.5,0.01,10000)


    # lists to store results
    total_trials_steps_Q = []
    total_trials_steps_SARSA = []
    total_trials_steps_MC = []

    total_returns_Q = []
    total_returns_SARSA = []
    total_returns_MC = []

    
    for i in trange(num_trials):
        print("TRAINING STARTED")
        each_episode_steps_Q, Q_table_Q, each_episode_return_Q = q_learning(env=tether_env,num_episodes=num_episodes,gamma=gamma,epsilon=epsilon, step_size=step_size)
        each_episode_steps_SARSA, Q_table_SARSA, each_episode_return_SARSA = SARSA(env=tether_env,num_episodes=num_episodes,gamma=gamma,epsilon=epsilon, step_size=step_size)
        each_episode_steps_MC, Q_table_MC, each_episode_return_MC = on_policy_mc_control_epsilon_soft(env=tether_env,num_episodes=num_episodes,gamma=gamma,epsilon=epsilon)
        
        total_trials_steps_Q.append(each_episode_steps_Q)
        total_trials_steps_SARSA.append(each_episode_steps_SARSA)
        total_trials_steps_MC.append(each_episode_steps_MC)

        total_returns_Q.append(each_episode_return_Q)
        total_returns_SARSA.append(each_episode_return_SARSA)
        total_returns_MC.append(each_episode_return_MC)
    
    legend_list = ["Q Learning", "SARSA", "Monte-Carlo"]

    # plotting learning curve
    total_steps_list = [total_trials_steps_Q,total_trials_steps_SARSA,total_trials_steps_MC]
    plot_learning_curve(total_steps_list, legend_list)

    # plotting return
    total_return_list = [total_returns_Q,total_returns_SARSA,total_returns_MC]
    plot_return(total_return_list, legend_list)

    # visualizing the trained model
    checkpoint_data = load_checkpoint('Q_checkpoint_best')
    Q_table = checkpoint_data['Q_table']
    test_algorithm(env=tether_env, epsilon=0, Q=Q_table)



if __name__ == "__main__":
    main()