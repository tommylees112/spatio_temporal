import torch.nn as nn
import torch
import torch.functional as F


class CustomLoss(nn.Module):
    """https://github.com/IgorSusmelj/pytorch-styleguide"""

    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, x, y):
        loss = torch.mean((x - y) ** 2)
        return loss


class RMSELoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, x, y):
        loss = torch.sqrt(F.mse_loss(x, y))
        return loss
