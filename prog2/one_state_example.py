import numpy as np

from prog2.env import Env, EnvSpec, EnvWithModel


class OneStateMDP(Env):  # MDP introduced at Fig 5.4 in Sutton Book
    def __init__(self):
        env_spec = EnvSpec(2, 2, 1.)

        super().__init__(env_spec)
        self.final_state = 1
        self.trans_mat, self.r_mat = self._build_trans_mat()

    def _build_trans_mat(self):
        trans_mat = np.zeros((2, 2, 2))

        trans_mat[0, 0, 0] = 0.9
        trans_mat[0, 0, 1] = 0.1
        trans_mat[0, 1, 0] = 0.
        trans_mat[0, 1, 1] = 1.0
        trans_mat[1, :, 1] = 1.

        r_mat = np.zeros((2, 2, 2))
        r_mat[0, 0, 1] = 1.

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


class OneStateMDPWithModel(OneStateMDP, EnvWithModel):
    @property
    def TD(self) -> np.array:
        return self.trans_mat

    @property
    def R(self) -> np.array:
        return self.r_mat