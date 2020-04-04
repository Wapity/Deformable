
import torch.nn as nn
import layers


class LinDecoder(nn.Module):
    def __init__(self, in_channels):
        super(LinDecoder, self).__init__()
        self._args = _args
        modules = [layers.GlobalPooling(),
                   layers.Conv3D(in_channels, 12, instance_norm=False, act='linear')]
        self._main = nn.Sequential(*modules)

    def forward(self, x):
        return self._main(x)


class DefDecoder(nn.Module):
    def __init__(self, in_channels):
        super(DefDecoder, self).__init__()
        self._args = _args
        modules = [layers.ChannelSELayer3D(in_channels),
                   layers.Conv3D(in_channels, 128),
                   layers.Conv3D(128, 64),
                   layers.Conv3D(64, 32),
                   layers.Conv3D(32, 32),
                   layers.Conv3D(32, 32),
                   layers.Conv3D(32, 3, instance_norm=False, act='sigmoid')]
        self._main = nn.Sequential(*modules)

    def forward(self, x):
        return self._main(x)
