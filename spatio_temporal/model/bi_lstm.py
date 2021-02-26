import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim


class BiLSTM(nn.Module):
    """Bi-Directional LSTM
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
    
    Args:
        nn ([type]): [description]
    """

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
        self.hidden_size = int(hidden_size)
        self.input_size = input_size
        self.output_size = output_size
        self.forecast_horizon = forecast_horizon
        self.dropout = nn.Dropout(p=dropout_rate)

        self.num_layers = 1

        #  LSTM cell
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        fc_layer = nn.Linear(hidden_size * 2, self.output_size)  # 2 for bidirection
        self.head = nn.Sequential(*[fc_layer])

        #  self.intialize_weights
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

        if np.product(data["x_f"].size()) > 0:
            # padding zero: https://stackoverflow.com/a/53126241/9940782
            #  NOTE: all assuming that batch_first !
            x_f = data["x_f"]
            new_dims = x_f.shape[1]
            target = torch.zeros(x_d.shape[0], x_f.shape[1], x_d.shape[-1])
            target[:, : x_d.shape[1], :] = x_d
            x_d = torch.cat([target, x_f], dim=-1)

        # Set initial states [1, batch_size, hidden_size]
        h0 = torch.zeros(self.num_layers * 2, x_d.size(0), self.hidden_size).to(
            x_d.device
        )
        c0 = torch.zeros(self.num_layers * 2, x_d.size(0), self.hidden_size).to(
            x_d.device
        )

        # Forward propagate LSTM
        # out: [batch_size, seq_length, hidden_size*2]
        lstm_output, (h_n, c_n) = self.lstm(x_d, (h0, c0))

        # final_output = [batch_size, 1, hidden_size]
        # only return the predictions from the final step in sequence_length
        final_output = lstm_output[:, -1:, :]
        y_hat = self.head(self.dropout(final_output))

        pred = {"h_n": h_n, "c_n": c_n, "y_hat": y_hat}
        return pred
