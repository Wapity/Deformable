
import torch.nn as nn
import layers


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self._cat = layers.concat()
        self._dilated_1 = layers.Conv3D(in_channels, 32, dilation=1)
        self._dilated_2 = layers.Conv3D(32, 64, dilation=1)
        self._dilated_3 = layers.Conv3D(64, 128, dilation=2)
        self._dilated_4 = layers.Conv3D(128, 32, dilation=3)
        self._dilated_5 = layers.Conv3D(32, 32, dilation=5)

    def forward(self, x):
        x = self._cat(x[0], x[1])
        x1 = self._dilated_1(x)
        x2 = self._dilated_2(x1)
        x3 = self._dilated_3(x2)
        x4 = self._dilated_4(x3)
        x5 = self._dilated_5(x4)
        x = torch.stack([x1, x2, x3, x4, x5], dim=1)
        return x
