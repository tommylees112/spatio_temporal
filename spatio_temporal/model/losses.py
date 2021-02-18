import torch.nn as nn
import torch
import torch.nn.functional as F


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

    def forward(self, x, y, epsilon: float = 1e-4):
        mse_loss = F.mse_loss(x, y)
        if mse_loss == 0:
            mse_loss = mse_loss + epsilon
        loss = torch.sqrt(mse_loss)
        return loss
