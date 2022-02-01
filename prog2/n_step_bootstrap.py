import itertools
from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy


def on_policy_n_step_td(
        env_spec: EnvSpec,
        trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
        n: int,
        alpha: float,
        initV: np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################

    V = initV.copy()
    for traj in trajs:
        T = 1 << 30
        G_acc = 0
        reward_arr = []
        state_arr = []
        traj_itr = iter(traj)
        for t in itertools.count():
            if t < T:
                try:
                    s, a, rn, sn = next(traj_itr)
                    reward_arr.append(rn)
                    if t == 0:
                        state_arr.append(s)
                    state_arr.append(sn)
                except StopIteration:
                    T = t
            tau = t - n + 1
            if tau > 0:
                G_acc -= reward_arr[tau - 1]
            G_acc /= env_spec.gamma
            if tau + n <= T:
                G_acc += np.power(env_spec.gamma, n - 1) * reward_arr[tau + n - 1]
            G = G_acc if tau + n >= T else G_acc + np.power(env_spec.gamma, n) * V[state_arr[tau + n]]
            if tau >= 0:
                V[state_arr[tau]] += alpha * (G - V[state_arr[tau]])
            if tau >= T - 1:
                break

    return V


def off_policy_n_step_sarsa(
        env_spec: EnvSpec,
        trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
        bpi: Policy,
        n: int,
        alpha: float,
        initQ: np.array
) -> Tuple[np.array, Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    #####################
    # Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################

    Q = initQ.copy()
    pi_mat = np.ones((env_spec.nS, env_spec.nA)) / env_spec.nA

    for traj in trajs:
        T = 1 << 30
        G_acc = 0
        rho_acc = 1
        reward_arr = []
        state_arr = []
        action_arr = []
        traj_itr = iter(traj)
        for t in itertools.count():
            if t < T:
                try:
                    s, a, rn, sn = next(traj_itr)
                    reward_arr.append(rn)
                    action_arr.append(a)
                    if t == 0:
                        state_arr.append(s)
                    state_arr.append(sn)
                except StopIteration:
                    T = t
                    break

        pi2bpi_arr = [pi_mat[state_arr[0], action_arr[0]] / bpi.action_prob(state_arr[0], action_arr[0])]
        for t in itertools.count():
            tau = t - n + 1
            if tau > 0:
                G_acc -= reward_arr[tau - 1]
            if tau >= 0:
                if np.isclose(pi2bpi_arr[tau], 0):
                    rho_acc = np.prod(pi2bpi_arr[tau + 1:min(tau + n, T - 1) + 1])
                else:
                    rho_acc /= pi2bpi_arr[tau]
            G_acc /= env_spec.gamma
            if t < T:
                G_acc += np.power(env_spec.gamma, n - 1) * reward_arr[tau + n - 1]
            if t + 1 <= T - 1:
                pi2bpi_arr.append(
                    pi_mat[state_arr[t + 1], action_arr[t + 1]] / bpi.action_prob(state_arr[t + 1], action_arr[t + 1]))
                rho_acc *= pi2bpi_arr[t + 1]

            G = G_acc if tau + n >= T else G_acc + np.power(env_spec.gamma, n) * Q[
                state_arr[tau + n], action_arr[tau + n]]
            if tau >= 0:
                Q[state_arr[tau], action_arr[tau]] += alpha * rho_acc * (G - Q[state_arr[tau], action_arr[tau]])

                # Update pi to make sure it is greedy
                s = state_arr[t]
                best_action = Q[s, :].argmax()
                if not np.isscalar(best_action):
                    best_action = best_action[0]
                pi_vec = np.zeros(env_spec.nA)
                pi_vec[best_action] = 1
                pi_mat[s, :] = pi_vec

                if tau >= T - 1:
                    break

    class MyPolicy(Policy):
        def __init__(self, pi_mat):
            self.pi_mat = pi_mat

        def action_prob(self, state: int, action: int) -> float:
            return self.pi_mat[state, action]

        def action(self, state: int) -> int:
            best_action = self.pi_mat[state, :].argmax()
            if np.isscalar(best_action):
                return best_action
            else:
                return best_action[0]

    pi = MyPolicy(pi_mat)

    return Q, pi
