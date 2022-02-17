from itertools import chain

import numpy as np
from torch.optim import Adam

from algo import ValueFunctionWithApproximation

import torch
import torch.nn.functional as F


class TwoHiddenLayerNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(TwoHiddenLayerNet, self).__init__()
        composite_dims = [input_dim, *hidden_dims, 1]
        self.hidden_layers = [torch.nn.Linear(composite_dims[i], composite_dims[i + 1]).double() for i in
                              range(len(hidden_dims) + 1)]

    def forward(self, x):
        for layer in self.hidden_layers[:-1]:
            x = F.relu(layer(x))
        return self.hidden_layers[-1](x)

    def my_parameters(self):
        return [{"params": layer.parameters()} for layer in self.hidden_layers]


class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self, state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        self.net = TwoHiddenLayerNet(state_dims, (32, 32))
        self.optimizer = Adam(self.net.my_parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.loss_fn = torch.nn.MSELoss()

    def __call__(self, s):
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.tensor(s).double()).detach().numpy().flatten()[0]

    def update(self, alpha, G, s_tau):
        self.net.train()

        loss = self.loss_fn(torch.tensor(G), self.net(torch.tensor(s_tau).double()))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
