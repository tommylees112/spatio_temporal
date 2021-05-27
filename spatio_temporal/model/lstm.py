import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from typing import Optional
# from spatio_temporal.model.base import BaseNN


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        forecast_horizon: int,
        dropout_rate: float = 0,
        initial_forget_bias: Optional[float] = None,
        FINAL_OUTPUT: bool = False,
    ):
        super().__init__()

        # hyperparameters
        self.hidden_size = int(hidden_size)
        self.input_size = input_size
        self.output_size = output_size
        self.forecast_horizon = forecast_horizon
        self.dropout = nn.Dropout(p=dropout_rate)
        self.FINAL_OUTPUT = FINAL_OUTPUT

        self.num_layers = 1

        #  LSTM cell
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
        )

        #  Fully connected layer with ReLU activation
        fc_layer = nn.Linear(self.hidden_size, self.output_size)
        # self.head = nn.Sequential(*[fc_layer, nn.ReLU()])
        self.head = nn.Sequential(*[fc_layer])

        # initialize weights
        self.initialize_weights()

        # forget gate bias to be ON (if +ve)
        self._reset_parameters(initial_forget_bias=initial_forget_bias)

    def _reset_parameters(self, initial_forget_bias: Optional[float]):
        """Special initialization of certain model weights."""
        if initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.hidden_size:2 * self.hidden_size] = initial_forget_bias

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

    def forward(self, data):
        """
        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
                - `h_n`: hidden state at the last time step of the sequence of shape [batch size, 1, hidden size].
                - `c_n`: cell state at the last time step of the sequence of shape [batch size, 1, hidden size].
        """
        # input = [batch_size, seq_length, n_features]
        x_d = data["x_d"]

        #  STATIC DATA
        if np.product(data["x_s"].size()) > 0:
            #  IN  [batch_size, n_static_features]
            # OUT [batch_size, seq_length, n_static_features]
            #  repeat for each seq_length and append new features to n_features
            x_s = data["x_s"].unsqueeze(1).repeat(1, x_d.shape[1], 1)

            #  concatenate onto x_d
            #   [batch_size, seq_length, all_features]
            x_d = torch.cat([x_d, x_s], dim=-1)

        if np.product(data["x_f"].size()) > 0:
            # padding zero: https://stackoverflow.com/a/53126241/9940782
            #  NOTE: all assuming that batch_first !
            #  [batch_size, seq_length, n_forecast_features]
            x_f = data["x_f"]
            new_seq_length = x_f.shape[1]
            #  create a new array of zeros to be filled
            #  [batch_size, seq_length, n_dynamic_features]
            target = torch.zeros(
                x_d.shape[0], new_seq_length, x_d.shape[-1], device=x_d.device
            )
            #  populate the values from the original dynamic data
            target[:, : x_d.shape[1], :] = x_d

            # concatenate the forecast variables as new columns
            x_d = torch.cat([target, x_f], dim=-1)

        # Set initial states [1, batch_size, hidden_size]
        h0 = torch.zeros(self.num_layers, x_d.size(0), self.hidden_size).to(x_d.device)
        c0 = torch.zeros(self.num_layers, x_d.size(0), self.hidden_size).to(x_d.device)

        #  lstm_output = (batch_size, seq_length, hidden_size)
        lstm_output, (h_n, c_n) = self.lstm(x_d, (h0, c0))

        if self.FINAL_OUTPUT:
            # final_output = [batch_size, 1, hidden_size]
            # only return the predictions from the final step in sequence_length
            final_output = lstm_output[:, -1:, :]
            y_hat = self.head(self.dropout(final_output))
        else:
            y_hat = self.head(self.dropout(lstm_output))

        pred = {"h_n": h_n, "c_n": c_n, "y_hat": y_hat}

        return pred
