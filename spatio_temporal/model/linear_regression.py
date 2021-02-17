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

        #  W and b parameters for optimizer to adjust
        self.W = nn.Parameter(self.W)
        self.b = nn.Parameter(self.b)

    def linear_model(self, x):
        return torch.matmul(self.dropout(x), self.W) + self.b

    def forward(self, data):
        x_d = data

        #  flatten all inputs
        x_d = x_d.view(-1, self.input_size)

        return {"y_hat": self.linear_model(x=x_d)}
