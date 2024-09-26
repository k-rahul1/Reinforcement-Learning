import gym
from typing import Optional, Callable
from collections import defaultdict
import numpy as np
from policy_v3 import create_epsilon_policy, create_scheduled_epsilon_policy
from tqdm import trange
import matplotlib.pyplot as plt
import pickle

def generate_episode(env: gym.Env, policy: Callable, es: bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state,_ = env.reset()
    while True:
        action = policy(state)

        next_state, reward, terminated, truncated, _ = env.step(action) 

        # state = state.astype(int)
        # next_state = next_state.astype(int)

        episode.append((state, action, reward))

        if terminated or truncated:
            break

        # print(f'state:{state} next state:{next_state} reward:{reward} action:{action} episode:{episode}')
        state = next_state

    return episode

# def q_learning(
#     env: gym.Env,
#     num_episodes: int,
#     gamma: float,
#     epsilon: float,
#     step_size: float,
# ):
#     """Q-learning

#     Args:
#         env (gym.Env): a Gym API compatible environment
#         num_episodes (int): Number of episodes
#         gamma (float): Discount factor of MDP
#         epsilon (float): epsilon for epsilon greedy
#         step_size (float): step size
#     """
#     Q = defaultdict(lambda: np.zeros(env.action_space.n))
#     time_steps_list = []

#     ep_bar = trange(num_episodes)
#     # for ep in ep_bar:
#     for ep in range(num_episodes):
#         state,_ = env.reset()
#         count_step =0
#         G = 0

#         while True:
#             count_step += 1

#             policy = create_epsilon_policy(Q, epsilon)
#             action = policy(state)
#             next_state, reward, terminated, truncated, info = env.step(action)

#             current_winding_angle = info['winding_angle']
            
#             # if count_step % 100 == 1:
#             #     print("reward",reward)

#             Q[state][action] += step_size * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

#             G += reward

#             if terminated or truncated:
#                 # print("final reward", reward)
#                 print("current winding angle",current_winding_angle)
#                 print("return",G)
#                 time_steps_list.append(count_step)
#                 break

#             state = next_state
#         # ep_bar.set_description(f"Episode: {ep} | Return: {G} | Steps: {count_step}")    

#     return time_steps_list, Q

# def test_q_learning(
#     env: gym.Env,
#     num_episodes: int,
#     gamma: float,
#     epsilon: float,
#     step_size: float,
#     Q:defaultdict
# ):
#     """Q-learning

#     Args:
#         env (gym.Env): a Gym API compatible environment
#         num_episodes (int): Number of episodes
#         gamma (float): Discount factor of MDP
#         epsilon (float): epsilon for epsilon greedy
#         step_size (float): step size
#     """
#     # Q = defaultdict(lambda: np.zeros(env.action_space.n))
#     time_steps_list = []

#     state,_ = env.reset()
#     count_step =0
#     path = []

#     while True:
#         count_step += 1

#         policy = create_epsilon_policy(Q, epsilon)
#         action = policy(state)
#         next_state, reward, terminated, truncated, _ = env.step(action)
#         path.append(state)
#         env.render(path=path)
#         # print("current pos",state)
#         # print("action taken",action)
#         # print("next pos",next_state)
#         # print("reward",reward)

#         # Q[state][action] += step_size * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

#         if terminated or truncated:
#             time_steps_list.append(count_step)
#             print("test Step",count_step)
#             break

#         state = next_state
            
#     return time_steps_list






def q_learning(
    env: gym.Env,
    num_episodes: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    Q = np.zeros((env.cols,env.rows,env.action_space.n))
    time_steps_list = []
    return_list = []

    ep_bar = trange(num_episodes)
    min_step = np.inf

    # for ep in range(num_episodes):
    for ep in ep_bar:
        state,_ = env.reset()
        count_step =0
        G = 0

        while True:
            # policy = create_epsilon_policy(Q, epsilon)
            policy = create_scheduled_epsilon_policy(Q, epsilon,ep)
            action = policy(state)
            # print("current state",state)
            # print("Action taken",action)
            next_state, reward, terminated, truncated, info = env.step(action)
            # print("next state",next_state)

            # current_winding_angle = next_state[2]
            
            # if count_step % 100 == 1:
            #     print("reward",reward)
            # state = state.astype(int)
            # next_state = next_state.astype(int)
            Q[state[0],state[1],action] += step_size * (reward + gamma * np.max(Q[next_state[0],next_state[1],:]) - Q[state[0],state[1],action])

            G += reward * (gamma**count_step)
            # print("return", G)

            count_step += 1
            
            # print("episode",ep, "steps", count_step, "current winding angle",current_winding_angle,"return",G )
            # env.render()

            if terminated or truncated:
                # print("final reward", reward)
                # print("current winding angle",current_winding_angle)
                # print("return",G)
                # print("episode",ep, "steps", count_step, "current winding angle",current_winding_angle,"return",G )
                time_steps_list.append(count_step)
                return_list.append(G)
                
                break

            state = next_state
        # Save checkpoint
        checkpoint_data = {'Q_table': Q}
        if (ep+1)/num_episodes == 0.25:
            with open('Q_checkpoint1', 'wb') as f:
                pickle.dump(checkpoint_data, f)
        elif (ep+1)/num_episodes == 0.5:
            with open('Q_checkpoint2', 'wb') as f:
                pickle.dump(checkpoint_data, f)
        elif (ep+1)/num_episodes == 0.75:
            with open('Q_checkpoint3', 'wb') as f:
                pickle.dump(checkpoint_data, f)
        elif (ep+1)/num_episodes == 1:
            with open('Q_checkpoint4', 'wb') as f:
                pickle.dump(checkpoint_data, f)

        if count_step < min_step:
            with open('Q_checkpoint_best', 'wb') as f:
                pickle.dump(checkpoint_data, f)
            min_step = count_step
        # ep_bar.set_description(f"Episode: {ep} | Return: {G} | Steps: {count_step} | winding angle: {current_winding_angle}")    
        ep_bar.set_description(f"Episode: {ep} | Return: {G} | Steps: {count_step}") 
    return time_steps_list, Q, return_list


def SARSA(
    env: gym.Env,
    num_episodes: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """SARSA algorithm.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """

    Q = np.zeros((env.cols,env.rows,env.action_space.n))
    time_steps_list = []
    return_list = []

    ep_bar = trange(num_episodes)
    # for ep in ep_bar:
    # for ep in range(num_episodes):
    for ep in ep_bar:
        state,_ = env.reset()
        # policy = create_epsilon_policy(Q, epsilon)
        policy = create_scheduled_epsilon_policy(Q, epsilon,ep)
        action = policy(state)

        count_step =0
        G = 0

        while True:  
            next_state, reward, terminated, truncated, info = env.step(action)
            # print("current state",state)
            # print("Action taken",action)
            # print("next state",next_state)

            # state = state.astype(int)
            # next_state = next_state.astype(int)

            # policy = create_epsilon_policy(Q, epsilon)
            policy = create_scheduled_epsilon_policy(Q, epsilon,ep)
            next_action = policy(next_state)

            # current_winding_angle = next_state[2]
            
            # if count_step % 100 == 1:
            #     print("reward",reward)

            Q[state[0],state[1],action] += step_size * (reward + gamma * Q[next_state[0],next_state[1],next_action] - Q[state[0],state[1],action])

            G += reward * (gamma**count_step)
            # print("return", G)

            count_step += 1
            
            # print("episode",ep, "steps", count_step, "current winding angle",current_winding_angle,"return",G )
            # env.render()

            if terminated or truncated:
                # print("final reward", reward)
                # print("current winding angle",current_winding_angle)
                # print("return",G)
                # print("episode",ep, "steps", count_step, "current winding angle",current_winding_angle,"return",G )
                time_steps_list.append(count_step)
                return_list.append(G)
                break

            state = next_state
            action = next_action
        # Save checkpoint
        checkpoint_data = {'Q_table': Q}
        if (ep+1)/num_episodes == 0.25:
            with open('SARSA_checkpoint1', 'wb') as f:
                pickle.dump(checkpoint_data, f)
        elif (ep+1)/num_episodes == 0.5:
            with open('SARSA_checkpoint2', 'wb') as f:
                pickle.dump(checkpoint_data, f)
        elif (ep+1)/num_episodes == 0.75:
            with open('SARSA_checkpoint3', 'wb') as f:
                pickle.dump(checkpoint_data, f)
        elif (ep+1)/num_episodes == 1:
            with open('SARSA_checkpoint4', 'wb') as f:
                pickle.dump(checkpoint_data, f) 

        # if count_step < min_step:
        #     with open('Q_checkpoint_best', 'wb') as f:
        #         pickle.dump(checkpoint_data, f)
        #     min_step = count_step       
        # ep_bar.set_description(f"Episode: {ep} | Return: {G} | Steps: {count_step} | winding angle: {current_winding_angle}")    
        ep_bar.set_description(f"Episode: {ep} | Return: {G} | Steps: {count_step}")    

    return time_steps_list, Q, return_list




def on_policy_mc_control_epsilon_soft(
    env: gym.Env, num_episodes: int, gamma: float, epsilon: float
):
    """On-policy Monte Carlo policy control for epsilon soft policies.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): Parameter for epsilon soft policy (0 <= epsilon <= 1)
    Returns:

    """
    Q = np.zeros((env.cols,env.rows,env.action_space.n))
    N = np.zeros((env.cols,env.rows,env.action_space.n))

    

    return_list = []

    time_steps_list = []
    min_step = np.inf

    ep_bar = trange(num_episodes)
    # for ep in ep_bar:
    # for ep in range(num_episodes):
    for ep in ep_bar:
        count_step =0
        G = 0
        # policy = create_epsilon_policy(Q, epsilon)
        policy = create_scheduled_epsilon_policy(Q, epsilon,ep)
        episode = generate_episode(env, policy)

        # current_winding_angle = episode[-1][0][2]

        for t in range(len(episode) - 1, -1, -1):   # t will start from T-1 and go to 0, with time step decrement by 1
            G = episode[t][2] + gamma * G           # calculating return backwards

            # Updating Q and N here according to first visit MC
            St = episode[t][0]
            At = episode[t][1]

            # print("state", St)

            N[St[0],St[1],At] += 1                          # incrementing the count for state St and action At pair
            Q[St[0],St[1],At] = Q[St[0],St[1],At] + (G-Q[St[0],St[1],At])/N[St[0],St[1],At]     # updating the Q value of (St,At) using increment average
            count_step += 1
        
        # print("episode",ep, "steps", count_step, "current winding angle",current_winding_angle,"return",G )
        # ep_bar.set_description(f"Episode: {ep} | Return: {G} | Steps: {count_step} | winding angle: {current_winding_angle}")
        ep_bar.set_description(f"Episode: {ep} | Return: {G} | Steps: {count_step}") 
        return_list.append(G)
        time_steps_list.append(count_step)
        # Save checkpoint
        checkpoint_data = {'Q_table': Q}
        if (ep+1)/num_episodes == 0.25:
            with open('MC_checkpoint1', 'wb') as f:
                pickle.dump(checkpoint_data, f)
        elif (ep+1)/num_episodes == 0.5:
            with open('MC_checkpoint2', 'wb') as f:
                pickle.dump(checkpoint_data, f)
        elif (ep+1)/num_episodes == 0.75:
            with open('MC_checkpoint3', 'wb') as f:
                pickle.dump(checkpoint_data, f)
        elif (ep+1)/num_episodes == 1:
            with open('MC_checkpoint4', 'wb') as f:
                pickle.dump(checkpoint_data, f)  

    return time_steps_list,Q,return_list


def test_algorithm(
    env: gym.Env,
    epsilon: float,
    Q
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    time_steps_list = []

    state,_ = env.reset()
    count_step =0
    path = []

    while True:
        count_step += 1

        policy = create_epsilon_policy(Q, epsilon)
        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        # state = state.astype(int)
        # next_state = next_state.astype(int) 
        env.anchor_list.update((state[0],state[1]),(next_state[0],next_state[1]),env.polyobstacles)
        anchors = env.anchor_list.anchors
        # for anch in anchors:
        #     print("current anchors",anch.pos)      
        path.append((state[0],state[1]))
        env.render(path=path, anchors=anchors)
        # print("current pos",state)
        # print("action taken",action)
        # print("next pos",next_state)

        if terminated or truncated:
            time_steps_list.append(count_step)
            print("test Step",count_step)
            break

        state = next_state
            
    return time_steps_list


def load_checkpoint(checkpoint_load_path):
    """Load checkpoint."""
    with open(checkpoint_load_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    return checkpoint_data


