import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pytorch_lightning as pl
from spatio_temporal.model.base_lightning import BaseModel


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.4,
    ):
        super().__init__()

        # hyperparameters
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
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
        # transpose to [seq_length, batch_size, n_features]
        x_d = data.transpose(0, 1)

        # if 'x_s' in data:
        #     x_s = data['x_s'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)

        lstm_output, (h_n, c_n) = self.lstm(input=x_d)

        # reshape to [batch_size, 1, n_hiddens]
        y_hat = self.head(self.dropout(lstm_output.transpose(0, 1)))

        h_n = h_n.transpose(0, 1)
        c_n = c_n.transpose(0, 1)

        pred = {"h_n": h_n, "c_n": c_n, "y_hat": y_hat}
        return pred
