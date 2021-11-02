from typing import List

import numpy as np
import torch
import torch.nn as nn


class FC(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], activation: str = 'tanh', dropout: float = 0.0):
        super(FC, self).__init__()

        activation = nn.Tanh()

        # create network from hidden_sizes
        layers = []
        if hidden_sizes:
            for i, hidden_size in enumerate(hidden_sizes):
                if i == 0:
                    layers.append(nn.Linear(input_size, hidden_size))
                else:
                    layers.append(nn.Linear(hidden_sizes[i - 1], hidden_size))

                layers.append(activation)
                layers.append(nn.Dropout(p=dropout))

            layers.append(nn.Linear(hidden_size, self.output_size))
        else:
            layers.append(nn.Linear(input_size, self.output_size))

        self.net = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        for layer in self.net:
            if isinstance(layer, nn.modules.linear.Linear):
                n_in = layer.weight.shape[1]
                gain = np.sqrt(3 / n_in)
                nn.init.uniform_(layer.weight, -gain, gain)
                nn.init.constant_(layer.bias, val=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
