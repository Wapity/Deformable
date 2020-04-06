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


class LinDecoder(nn.Module):

    def __init__(self, in_channels):
        super(LinDecoder, self).__init__()
        modules = [
            nn.AdaptiveAvgPool3d((None, None, None, 1, 1)),
            Conv3D(in_channels, 12, instance_norm=False)
        ]
        self._main = nn.Sequential(*modules)

    def forward(self, x):
        x = self._main(x)
        print(x.shape)
        return x


class DefDecoder(nn.Module):

    def __init__(self, in_channels):
        super(DefDecoder, self).__init__()
        modules = [
            ChannelSELayer3D(in_channels),
            Conv3D(in_channels, 128),
            Conv3D(128, 64),
            Conv3D(64, 32),
            Conv3D(32, 32),
            Conv3D(32, 32),
            Conv3D(32, 3, instance_norm=False, act='sigmoid')
        ]
        self._main = nn.Sequential(*modules)

    def forward(self, x):
        return self._main(x) / 2
