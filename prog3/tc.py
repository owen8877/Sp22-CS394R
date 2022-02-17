import numpy as np
from algo import ValueFunctionWithApproximation


class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low: np.array,
                 state_high: np.array,
                 num_tilings: int,
                 tile_width: np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        num_tilings_in_dim = (np.ceil((state_high - state_low) / tile_width) + 1).astype(np.int)
        self.weights = np.zeros((num_tilings, *num_tilings_in_dim))

        self.state_high = state_high
        self.state_low = state_low
        self.tile_width = tile_width
        self.num_tilings = num_tilings

    def _coordinates(self, s):
        # dim=#0, tiling=#1
        origins = self.state_low[:, np.newaxis] - (np.arange(self.num_tilings) / self.num_tilings)[np.newaxis,
                                                  :] * self.tile_width[:, np.newaxis]
        return np.floor((s[:, np.newaxis] - origins) / self.tile_width[:, np.newaxis]).astype(np.int)

    def __call__(self, s):
        coordinates = self._coordinates(s)
        return sum((self.weights[(i, *coordinates[:, i])] for i in range(self.num_tilings)))

    def update(self, alpha, G, s_tau):
        coordinates = self._coordinates(s_tau)
        v_hat = sum((self.weights[(i, *coordinates[:, i])] for i in range(self.num_tilings)))
        update_val = alpha * (G - v_hat)
        for i in range(self.num_tilings):
            self.weights[(i, *coordinates[:, i])] += update_val
