import sys
from functools import reduce
from typing import Tuple

import numpy as np


def sample_average_update(value_estimates, selected_counts, selected_arm, sample_value):
    selected_counts[selected_arm] += 1
    n = selected_counts[selected_arm]
    value_estimates[selected_arm] += (sample_value - value_estimates[selected_arm]) / n


def constant_step_size_update(value_estimates, selected_counts, selected_arm, sample_value, alpha: float = 0.1):
    selected_counts[selected_arm] += 1
    value_estimates[selected_arm] += (sample_value - value_estimates[selected_arm]) * alpha


def testbed(update_strategy, N_itr: int = 10000, N_test: int = 300, N_arm: int = 10, eps: float = 0.1,
            delta_true_value_change: float = 0.01, std_arm_distribution: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    # Set-up tracking quantities
    avg_rewards_list = []
    ratio_optimal_actions_list = []

    # Set-up true expected rewards for each arm
    for i in range(N_test):
        values = np.zeros((N_arm,))
        value_estimates = np.zeros((N_arm,))
        selected_counts = np.zeros((N_arm,))
        acc_reward = 0
        acc_rewards_list = []
        is_optimal_list = []

        # Pre-generate all random numbers
        value_change_rand_pool = np.random.randn(N_itr, N_arm)
        sample_rand_pool = np.random.randn(N_itr)
        eps_decision_rand_pool = np.random.rand(N_itr)

        for t in range(N_itr):
            # Update the true expected value
            # values += np.random.randn(N_arm) * delta_true_value_change
            values += value_change_rand_pool[t, :] * delta_true_value_change

            # Determine first if to explore or exploit
            # if np.random.rand(1) < eps:
            if eps_decision_rand_pool[t] < eps:
                # Explore time!
                selected_arm = np.random.randint(0, N_arm)
            else:
                # Exploit the current optimal arm
                selected_arms = np.argmax(value_estimates)
                if not np.isscalar(selected_arms):
                    # There is a tie and we randomly pick one
                    selected_arm = selected_arms[np.random.randint(0, len(selected_arms))]
                else:
                    selected_arm = selected_arms

            # Draw a sample from the selected arm
            # sample_value = np.random.randn(1)[0] * std_arm_distribution + values[selected_arm]
            sample_value = sample_rand_pool[t] * std_arm_distribution + values[selected_arm]

            # Update the estimates
            update_strategy(value_estimates, selected_counts, selected_arm, sample_value)

            # Update tracking quantities
            if values[selected_arm] == values.max():
                is_optimal_list.append(1)
            else:
                is_optimal_list.append(0)

            acc_reward += sample_value
            acc_rewards_list.append(acc_reward)

        avg_rewards_list.append(acc_rewards_list / np.arange(1, N_itr + 1))
        ratio_optimal_actions_list.append(is_optimal_list)

    # Average tracking quantities
    avg_rewards = np.average(avg_rewards_list, axis=0)
    ratio_optimal_actions = np.average(ratio_optimal_actions_list, axis=0)
    return avg_rewards, ratio_optimal_actions


if __name__ == '__main__':
    # Run test and output file
    filename = sys.argv[1]
    sample_avg_result = testbed(update_strategy=sample_average_update)
    constant_step_result = testbed(update_strategy=constant_step_size_update)
    np.savetxt(filename, np.array([*sample_avg_result, *constant_step_result]))
