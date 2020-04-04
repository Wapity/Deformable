import torch
import torch.nn as nn

activations = {'relu': nn.ReLU(),
               'tanh': nn.Tanh(),
               'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
               'sigmoid': nn.Sigmoid()}


def GlobalPooling(final=(1, 1, 1)):
    return nn.AdaptiveAvgPool3d(final)


class Conv3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 instance_norm=True,
                 act='leaky_relu'):
        super(Conv3D, self).__init__()
        self._layer = nn.Conv3d(in_channels, out_channels,
                                kernel_size, stride, padding, dilation)
        if instance_norm:
            self._norm = nn.InstanceNorm3d(
                in_channels, eps=1e-05, momentum=0.1)
        else:
            self._norm = nn.Identity()
        self._act = activations[act]

    def forward(self, x):
        print('input', x.shape)
        x = self._layer(x)
        x = self._norm(x)
        x = self._act(x)
        print('output', x.shape)
        return x


class ChannelSELayer3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        """
        :param reduction_ratio: By how much should the in_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self._avg_pool = nn.AdaptiveAvgPool3d(1)
        in_channels_reduced = in_channels // reduction_ratio
        self._reduction_ratio = reduction_ratio
        self._fc1 = nn.Linear(in_channels, in_channels_reduced, bias=True)
        self._fc2 = nn.Linear(in_channels_reduced, in_channels, bias=True)
        self._relu = nn.ReLU()
        self._sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        :param x: X, shape = (batch_size, in_channels, D, H, W)
        """
        batch_size, in_channels, D, H, W = x.size()
        # Average along each channel
        squeeze_tensor = self._avg_pool(x)

        # channel excitation
        fc_out_1 = self._relu(
            self._fc1(squeeze_tensor.view(batch_size, in_channels)))
        fc_out_2 = self._sigmoid(self._fc2(fc_out_1))

        output_tensor = torch.mul(x, fc_out_2.view(
            batch_size, in_channels, 1, 1, 1))

        return output_tensor
