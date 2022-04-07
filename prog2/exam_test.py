import numpy as np

from dp import value_prediction
from prog2.env import EnvWithModel, Env, EnvSpec
from prog2.policy import Policy


class RectModel_(Env):  # MDP introduced at Fig 5.4 in Sutton Book
    def __init__(self):
        env_spec = EnvSpec(12, 3, 1.)

        super().__init__(env_spec)
        self.final_state = 1
        self.trans_mat, self.r_mat = self._build_trans_mat()

    def _build_trans_mat(self):
        trans_mat = np.zeros((12, 3, 12))

        trans_mat[0, 0, 0] = 1.
        trans_mat[0, 1, 1] = 1.
        trans_mat[0, 2, 4] = 1.
        trans_mat[1, 0, 1] = 1.
        trans_mat[1, 1, 2] = 1.
        trans_mat[1, 2, 5] = 1.
        trans_mat[2, 0, 2] = 1.
        trans_mat[2, 1, 3] = 1.
        trans_mat[2, 2, 6] = 1.
        trans_mat[3, 0, 3] = 1.
        trans_mat[3, 1, 3] = 1.
        trans_mat[3, 2, 7] = 1.
        trans_mat[4, 0, 0] = 1.
        trans_mat[4, 1, 5] = 1.
        trans_mat[4, 2, 8] = 1.
        trans_mat[5, 0, 1] = 1.
        trans_mat[5, 1, 6] = 1.
        trans_mat[5, 2, 9] = 1.
        trans_mat[6, 0, 2] = 1.
        trans_mat[6, 1, 7] = 1.
        trans_mat[6, 2, 10] = 1.
        trans_mat[7, :, 7] = 1.
        trans_mat[8, 0, 4] = 1.
        trans_mat[8, 1, 9] = 1.
        trans_mat[8, 2, 8] = 1.
        trans_mat[9, 0, 5] = 1.
        trans_mat[9, 1, 10] = 1.
        trans_mat[9, 2, 9] = 1.
        trans_mat[10, 0, 6] = 1.
        trans_mat[10, 1, 11] = 1.
        trans_mat[10, 2, 10] = 1.
        trans_mat[11, 0, 7] = 1.
        trans_mat[11, 1, 11] = 1.
        trans_mat[11, 2, 11] = 1.

        # trans_mat[0, 0, 0] = 0.9
        # trans_mat[0, 0, 1] = 0.1
        # trans_mat[0, 1, 0] = 0.
        # trans_mat[0, 1, 1] = 1.0
        # trans_mat[1, :, 1] = 1.

        r_mat = np.zeros((12, 3, 12))
        r_mat[0, 0, 0] = -1.
        r_mat[1, 0, 1] = -1.
        r_mat[2, 0, 2] = -1.
        r_mat[3, 0, 3] = -1.
        r_mat[3, 1, 3] = -1.
        r_mat[3, 2, 7] = 100.
        r_mat[6, 1, 7] = 100.
        r_mat[8, 2, 8] = -1.
        r_mat[9, 2, 9] = -1.
        r_mat[10, 2, 10] = -1.
        r_mat[11, 0, 7] = 100.
        r_mat[11, 1, 11] = -1.
        r_mat[11, 2, 11] = -1.

        return trans_mat, r_mat

    def reset(self):
        self._state = 0
        return self._state

    def step(self, action):
        assert action in list(range(self.spec.nA)), "Invalid Action"
        assert self._state != self.final_state, "Episode has ended!"

        prev_state = self._state
        self._state = np.random.choice(self.spec.nS, p=self.trans_mat[self._state, action])
        r = self.r_mat[prev_state, action, self._state]

        if self._state == self.final_state:
            return self._state, r, True
        else:
            return self._state, r, False


class RectModel(RectModel_, EnvWithModel):
    @property
    def TD(self) -> np.array:
        return self.trans_mat

    @property
    def R(self) -> np.array:
        return self.r_mat


class RandomPolicy(Policy):
    def __init__(self, nA, p=None):
        self.p = p if p is not None else np.array([1 / nA] * nA)

    def action_prob(self, state, action=None):
        return self.p[action]

    def action(self, state):
        return np.random.choice(len(self.p), p=self.p)


env = RectModel()
V, _ = value_prediction(env, RandomPolicy(env.spec.nA), np.zeros(env.spec.nS), 1e-12)
print(V)
