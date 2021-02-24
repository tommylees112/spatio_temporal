import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim


class BaseNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        forecast_horizon: int,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        # hyperparameters
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.forecast_horizon = forecast_horizon
        self.dropout = nn.Dropout(p=dropout_rate)

    def initialize_weights(self):
        # We are initializing the weights here with Xavier initialisation
        #  (by multiplying with 1/sqrt(n))
        # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        sqrt_k = np.sqrt(1 / self.hidden_size)
        for parameters in self.lstm.parameters():
            for pam in parameters:
                nn.init.uniform_(pam.data, -sqrt_k, sqrt_k)

        for dense_layer in self.head:
            if not isinstance(dense_layer, nn.ReLU):
                nn.init.kaiming_uniform_(dense_layer.weight.data)
                nn.init.constant_(dense_layer.bias.data, 0)
