import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from spatio_temporal.model.base_lightning import BaseModel


class LSTM(nn.Module):
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

        #  LSTM cell
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )

        #  Fully connected layer with ReLU activation
        fc_layer = nn.Linear(hidden_size, output_size)
        # self.head = nn.Sequential(*[fc_layer, nn.ReLU()])
        self.head = nn.Sequential(*[fc_layer])

        # initialize weights
        self.initialize_weights()

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

    def _return_forecast_times(self, y_hat):
        # [batch_size, seq_length, forecast_horizon]
        #  forecast_horizon = number of target timesteps
        return y_hat[:, -1:, :]

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
        x_d = data

        # if 'x_s' in data:
        #     x_s = data['x_s'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
        #     # concatenate onto x_d

        #  output = (batch_size, seq_length, hidden_size)
        lstm_output, (h_n, c_n) = self.lstm(input=x_d)

        # output = [batch_size, seq_length, 1]
        # only return the predictions from the final step in sequence_length
        y_hat = self.head(self.dropout(lstm_output))
        y_hat = y_hat[:, -1:, :]

        pred = {"h_n": h_n, "c_n": c_n, "y_hat": y_hat}
        return pred
