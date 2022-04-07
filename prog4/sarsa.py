import numpy as np
from tqdm import trange


class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low: np.array,
                 state_high: np.array,
                 num_actions: int,
                 num_tilings: int,
                 tile_width: np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.num_tiles_in_dim = (np.ceil((state_high - state_low) / tile_width) + 1).astype(np.int)
        self.state_high = state_high
        self.state_low = state_low
        self.tile_width = tile_width
        self.num_tilings = num_tilings
        self.num_actions = num_actions

        self.origins = self.state_low[:, np.newaxis] - (np.arange(self.num_tilings) / self.num_tilings)[np.newaxis,
                                                       :] * self.tile_width[:, np.newaxis]

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.num_actions * self.num_tilings * np.prod(self.num_tiles_in_dim)

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        if done:
            return np.zeros((self.feature_vector_len(),))
        else:
            coordinates = np.floor((s[:, np.newaxis] - self.origins) / self.tile_width[:, np.newaxis]).astype(np.int)
            feature = np.zeros((self.num_actions, self.num_tilings, *self.num_tiles_in_dim))
            for i in range(self.num_tilings):
                feature[(a, i, *coordinates[:, i])] = 1
            return feature.flatten()


def SarsaLambda(
        env,  # openai gym environment
        gamma: float,  # discount factor
        lam: float,  # decay rate
        alpha: float,  # step size
        X: StateActionFeatureVectorWithTile,
        num_episode: int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s, done, w, epsilon=.01):
        nA = env.action_space.n
        Q = [np.dot(w, X(s, done, a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len(),))
    for _ in trange(num_episode):
        s, done = env.reset(), False
        a = epsilon_greedy_policy(s, done, w)
        x = X(s, done, a)
        z = np.zeros(w.shape)
        Q_old = 0
        while not done:
            s, r, done, _ = env.step(a)
            a = epsilon_greedy_policy(s, done, w)
            x_ = X(s, done, a)
            Q = np.dot(w, x)
            Q_ = np.dot(w, x_)
            delta = r + gamma * Q_ - Q
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
            w += alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = Q_
            x = x_

    return w
