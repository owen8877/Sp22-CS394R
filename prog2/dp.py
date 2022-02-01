from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy


def value_prediction(env: EnvWithModel, pi: Policy, initV: np.array, theta: float) -> Tuple[np.array, np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # Implement Value Prediction Algorithm (Hint: Sutton Book p.75)
    #####################

    # Cache the action prob matrix
    action_prob_mat = np.array([[pi.action_prob(s, a) for s in range(env.spec.nS)] for a in range(env.spec.nA)])

    # Init optimization variables
    V = initV.copy()

    # Loop to update the value prediction
    while True:
        Delta = 0  # maximal update error
        for s in range(env.spec.nS):
            v = V[s]
            pi_vec = action_prob_mat[:, s]
            p_mat = env.TD[s, :, :]
            r_mat = env.R[s, :, :]
            return_mat = pi_vec[:, np.newaxis] * p_mat * (r_mat + env.spec.gamma * V[np.newaxis, :])
            V[s] = return_mat.sum()
            Delta = max(Delta, np.abs(V[s] - v))

        if Delta < theta:
            break

    # Now use the value prediction to estimate action-value
    Q = (env.TD * (env.R + env.spec.gamma * V[np.newaxis, np.newaxis, :])).sum(axis=-1)

    return V, Q


def value_iteration(env: EnvWithModel, initV: np.array, theta: float) -> Tuple[np.array, Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    #####################
    # Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################

    # Init optimization variables
    V = initV.copy()

    # Loop to update the value prediction of optimal policy
    while True:
        Delta = 0  # maximal update error
        for s in range(env.spec.nS):
            v = V[s]
            p_mat = env.TD[s, :, :]
            r_mat = env.R[s, :, :]
            return_mat = (p_mat * (r_mat + env.spec.gamma * V[np.newaxis, :])).sum(axis=1).max()
            V[s] = return_mat.sum()
            Delta = max(Delta, np.abs(V[s] - v))

        if Delta < theta:
            break

    # Compute the optimal policy that yields the value estimate
    class MyPolicy(Policy):
        def __init__(self, pi_mat):
            self.pi_mat = pi_mat

        def action_prob(self, state: int, action: int) -> float:
            return self.pi_mat[state, action]

        def action(self, state: int) -> int:
            return self.pi_mat[state, :].argmax()

    pi_mat = (env.TD * (env.R + env.spec.gamma * V[np.newaxis, np.newaxis, :])).sum(axis=2)
    pi = MyPolicy(pi_mat)

    return V, pi
