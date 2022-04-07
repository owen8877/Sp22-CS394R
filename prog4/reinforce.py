from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.adam import Adam
from tqdm import trange


class HiddenLayerNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(HiddenLayerNet, self).__init__()
        composite_dims = [input_dim, *hidden_dims, output_dim]
        self.hidden_layers = [torch.nn.Linear(composite_dims[i], composite_dims[i + 1]).double() for i in
                              range(len(hidden_dims) + 1)]

    def forward(self, x):
        for layer in self.hidden_layers[:-1]:
            x = F.relu(layer(x))
        return self.hidden_layers[-1](x)

    def my_parameters(self):
        return [{"params": layer.parameters()} for layer in self.hidden_layers]


class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        self.net = HiddenLayerNet(state_dims, (32, 32), num_actions)
        self.optimizer = Adam(self.net.my_parameters(), lr=alpha, betas=(0.9, 0.999))

    def __call__(self, s) -> int:
        s = torch.tensor(s).double()
        self.net.eval()
        with torch.no_grad():
            preference = self.net(s).detach().numpy()
        probability = np.exp(preference)
        probability /= probability.sum()
        rand = np.random.rand(1)
        return np.argmax(np.cumsum(probability) > rand[0])

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        s = torch.tensor(s).double()
        self.net.train()

        preference = self.net(s)
        probability = torch.exp(preference)
        prob_normal = probability / probability.sum()
        loss = - torch.log(prob_normal[a]) * gamma_t * delta
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """

    def __init__(self, b):
        self.b = b

    def __call__(self, s) -> float:
        return self.b

    def update(self, s, G):
        pass


class VApproximationWithNN(Baseline):
    def __init__(self, state_dims, alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        super().__init__(0)
        self.net = HiddenLayerNet(state_dims, (32, 32), 1)
        self.optimizer = Adam(self.net.my_parameters(), lr=alpha, betas=(0.9, 0.999))
        self.loss_fn = torch.nn.MSELoss()

    def __call__(self, s) -> float:
        s = torch.tensor(s).double()
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.tensor(s).double()).detach().numpy().flatten()[0]

    def update(self, s, delta):
        s = torch.tensor(s).double()
        self.net.train()

        loss = self.loss_fn(torch.tensor(delta), self.net(torch.tensor(s).double()))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def REINFORCE(
        env,  # open-ai environment
        gamma: float,
        num_episodes: int,
        pi: PiApproximationWithNN,
        V: Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    G_0s = []
    for _ in trange(num_episodes):
        sar_arr = []

        s, done = env.reset(), False
        while not done:
            a = pi(s)
            s_, r, done, _ = env.step(a)
            sar_arr.append((s, a, r))
            s = s_

        T = len(sar_arr)
        G_arr = np.zeros((T,))
        G_arr[T - 1] = sar_arr[T - 1][2]
        for t in range(T - 2, -1, -1):
            G_arr[t] = gamma * G_arr[t + 1] + sar_arr[t][2]

        for t, (s, a, r) in enumerate(sar_arr):
            delta = G_arr[t] - V(s)
            pi.update(s, a, np.power(gamma, t), delta)
            V.update(s, delta)

        G_0s.append(G_arr[0])
    return G_0s
