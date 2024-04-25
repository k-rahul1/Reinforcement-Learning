# Author
# Rahul Kumar (Northeastern University)

import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(step_count_list,legend_list):
    """
    Plots the learning curve based on the step counts per episode for multiple experiments.

    Parameters:
    - step_count_list (list): List containing arrays of step counts per episode for each experiment.
    - legend_list (list): List of legends for the experiments.

    Returns:
    None

    - Computes the average step count and standard error across experiments.
    - Plots the average step count along with the standard error as shaded regions.
    """
    for i,step_count in enumerate(step_count_list):
        average_step = np.average(step_count,axis=0)
        standard_error = np.std(step_count,axis=0)
        error = 1.96 * standard_error/np.sqrt(len(step_count))

        x = np.arange(len(step_count[0]))
        plt.plot(average_step,label=legend_list[i])
        plt.fill_between(x, (average_step - error), (average_step + error), alpha=0.4)

    plt.xlabel('Episodes')
    plt.ylabel('Steps per episode')
    plt.title('Learining curves')
    plt.legend()
    plt.show()

def plot_return(returns_list,legend_list):
    """
    Plots the return per episode for multiple experiments.

    Parameters:
    - returns_list (list): List containing arrays of returns per episode for each experiment.
    - legend_list (list): List of legends for the experiments.

    Returns:
    None

    - Computes the average return and standard error across experiments.
    - Plots the average return along with the standard error as shaded regions.
    """
    for i,returns in enumerate(returns_list):
        average_return = np.average(returns,axis=0)
        standard_error = np.std(returns,axis=0)
        error = 1.96 * standard_error/np.sqrt(len(returns))

        x = np.arange(len(returns[0]))
        plt.plot(average_return,label=legend_list[i])
        plt.fill_between(x, (average_return - error), (average_return + error), alpha=0.4)
    plt.xlabel('Episodes')
    plt.ylabel('Return per episode')
    plt.title('Comparison of Returns')
    plt.legend()
    plt.show()
