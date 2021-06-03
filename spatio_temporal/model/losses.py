import torch.nn as nn
import torch
import torch.nn.functional as F


# TODO: implement subsetting in time (i.e. only calculate loss on the final timestep)
# TODO: talk to Freddy & Daniel re: subsetting in time
# https://github.com/neuralhydrology/neuralhydrology/blob/ffbc60043dec4e1cfc1634cd70917291c1ee77be/neuralhydrology/training/loss.py#L125-L131

class CustomLoss(nn.Module):
    """https://github.com/IgorSusmelj/pytorch-styleguide"""

    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, x, y):
        loss = torch.mean((x - y) ** 2)
        return loss


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true, epsilon: float = 1e-4):
        mask = ~torch.isnan(y_true)
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        mse_loss = F.mse_loss(y_pred, y_true)
        if mse_loss == 0:
            mse_loss = mse_loss + epsilon
        loss = torch.sqrt(mse_loss)
        return loss


class NSELoss(nn.Module):
    """Calculate (batch-wise) NSE Loss.
    Each sample i is weighted by 1 / (std_i + eps)^2, where std_i is the standard deviation of the
    discharge from the basin, to which the sample belongs.
    Parameters:
    -----------
    eps : float
        Constant, added to the weight for numerical stability and smoothing, default to 0.1

    segmentation fault:
    https://github.com/pytorch/pytorch/issues/926
    """

    def __init__(self, eps: float = 0.1):
        super(NSELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, q_stds: torch.Tensor):
        """Calculate the batch-wise NSE Loss function.
        Parameters
        ----------
        y_pred : torch.Tensor
            Tensor containing the network prediction.
        y_true : torch.Tensor
            Tensor containing the true discharge values
        q_stds : torch.Tensor
            Tensor containing the discharge std (calculate over training period) of each sample
        Returns
        -------
        torch.Tenor
            The (batch-wise) NSE Loss
        """
        # [batch_size, seq_length, 1]
        mask = ~torch.isnan(y_true)
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        squared_error = (y_pred - y_true) ** 2
        weights = 1 / (q_stds + self.eps) ** 2
        scaled_loss = weights * squared_error

        return torch.mean(scaled_loss)

