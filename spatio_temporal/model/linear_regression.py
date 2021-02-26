import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class LinearRegression(nn.Module):
    """
    ```python
        model = LinearRegression(
            input_size=dl.input_size * cfg.seq_length,
            output_size=dl.output_size,
            forecast_horizon=cfg.horizon,
        ).to(cfg.device)
    ```
    """

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

        self.linear = nn.Linear(
            in_features=self.input_size, out_features=self.output_size
        )

    def forward(self, data):
        # input = [batch_size, seq_length, n_features]
        x_d = data["x_d"]

        if np.product(data["x_s"].size()) > 0:
            #  IN  [batch_size, n_static_features]
            # OUT [batch_size, seq_length, n_static_features]
            #  repeat for each seq_length and append new features to n_features
            x_s = data["x_s"].unsqueeze(1).repeat(1, x_d.shape[1], 1)

            #  concatenate onto x_d
            #   [batch_size, seq_length, all_features]
            x_d = torch.cat([x_d, x_s], dim=-1)

        #  flatten all inputs
        #  [batch_size, seq_length, n_features] -> [batch_size, self.input_size]
        # TODO: fix to work more generally
        x_d = x_d.view(-1, self.input_size)

        return {"y_hat": self.linear(x_d)}
