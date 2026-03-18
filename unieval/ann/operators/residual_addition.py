import torch
import torch.nn as nn

class ResidualAddition(nn.Module):
    def __init__(self):
        super(ResidualAddition, self).__init__()

    def forward(self, x1, x2):
        return x1 + x2


