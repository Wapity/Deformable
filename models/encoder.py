import torch
import torch.nn as nn
from .layers import *


class Encoder(nn.Module):

    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self._dilated_1 = Conv3D(in_channels, 32, 3, 1, dilation=1)
        self._dilated_2 = Conv3D(32, 64, 3, 1, dilation=1)
        self._dilated_3 = Conv3D(64, 128, 3, 1, dilation=2)
        self._dilated_4 = Conv3D(128, 32, 3, 1, dilation=3)
        self._dilated_5 = Conv3D(32, 32, 3, 1, dilation=5)

    def forward(self, x_0, x_1):
        x = torch.stack([x_0, x_1], 1)
        x1 = self._dilated_1(x)
        x2 = self._dilated_2(x1)
        x3 = self._dilated_3(x2)
        x4 = self._dilated_4(x3)
        x5 = self._dilated_5(x4)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return x
