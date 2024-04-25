# Author
# Rahul Kumar (Northeastern University)

import numpy as np
from collections import defaultdict
from typing import Callable, Tuple
from env_v5 import Action
import random

def create_epsilon_policy(Q, epsilon):
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = len(Q[0,0,0])

    def get_action(state):   
        if np.random.random() < epsilon:
            action = random.choice(range(0, num_actions))
        else:
            if state[2]>10:
                angle = 10
            else:
                angle = state[2]
            action = np.random.choice(np.where(Q[state[0],state[1],angle,:] == np.max(Q[state[0],state[1],angle,:]))[0])
        return action

    return get_action


def create_scheduled_epsilon_policy(Q, epsilon,steps):
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = len(Q[0,0,0])

    def get_action(state):   
        if np.random.random() < epsilon.value(steps):
            action = random.choice(range(0, num_actions))
        else:
            

            if state[2]>10:
                angle = 10
            else:
                angle = state[2]
            action = np.random.choice(np.where(Q[state[0],state[1],angle,:] == np.max(Q[state[0],state[1],angle,:]))[0])
        return action

    return get_action