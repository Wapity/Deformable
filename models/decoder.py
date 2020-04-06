import torch.nn as nn
from .layers import *


class LinDecoder(nn.Module):

    def __init__(self, in_channels):
        super(LinDecoder, self).__init__()
        modules = [
            nn.AdaptiveAvgPool3d((None, None, None, 1, 1)),
            layers.Conv3D(in_channels, 12, instance_norm=False)
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
            layers.ChannelSELayer3D(in_channels),
            layers.Conv3D(in_channels, 128),
            layers.Conv3D(128, 64),
            layers.Conv3D(64, 32),
            layers.Conv3D(32, 32),
            layers.Conv3D(32, 32),
            layers.Conv3D(32, 3, instance_norm=False, act='sigmoid')
        ]
        self._main = nn.Sequential(*modules)

    def forward(self, x):
        return self._main(x) / 2
