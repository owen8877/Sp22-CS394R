import numpy as np
from tqdm import trange

from policy import Policy


class ValueFunctionWithApproximation(object):
    def __call__(self, s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self, alpha, G, s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()


def semi_gradient_n_step_td(
        env,  # open-ai environment
        gamma: float,
        pi: Policy,
        n: int,
        alpha: float,
        V: ValueFunctionWithApproximation,
        num_episode: int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """

    for i in trange(num_episode):
        T = 1 << 30
        G_acc = 0
        reward_arr = []
        state_arr = []
        t = 0

        state, done = env.reset(), False
        state_arr.append(state)
        # env.render()

        while True:
            if t < T:
                a = pi.action(state)
                state, r, done, info = env.step(a)
                state_arr.append(state)
                reward_arr.append(r)
                # env.render()

                if done:
                    T = t + 1

            tau = t - n + 1
            if tau > 0:
                G_acc -= reward_arr[tau - 1]
            G_acc /= gamma
            if tau + n <= T:
                G_acc += np.power(gamma, n - 1) * reward_arr[tau + n - 1]
            G = G_acc if tau + n >= T else G_acc + np.power(gamma, n) * V(state_arr[tau + n])
            if tau >= 0:
                V.update(alpha, G, state_arr[tau])
            t += 1

            if tau == T - 1:
                break
