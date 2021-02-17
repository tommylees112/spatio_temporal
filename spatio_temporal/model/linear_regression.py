import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class LinearRegression(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        forecast_horizon: int,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0,
    ):
        super().__init__()

        # hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.forecast_horizon = forecast_horizon
        self.dropout = nn.Dropout(p=dropout_rate)

        self.W = Variable(
            torch.randn(self.input_size, self.output_size), requires_grad=True
        )
        self.b = Variable(torch.randn(self.output_size), requires_grad=True)

    def linear_model(self, x):
        return torch.matmul(self.dropout(x), self.W) + self.b

    def forward(self, data):
        x_d = data

        return self.linear_model(x=x_d)
