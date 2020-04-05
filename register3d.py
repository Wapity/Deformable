import torch.nn as nn
from layers import Transformer3d
from decoder import DefDecoder, LinDecoder
from encoder import Encoder
from utils import affine_grid_3d, gradient_grid_3d
from utils import build_affine_grid


class Register3d(nn.Module):
    def __init__(self, size, device, linear=True):
        super(Register3d, self).__init__()
        self.encoder = Encoder()
        self.def_decoder = DefDecoder()
        self.a_decoder = LinDecoder
        self.transformer = Transformer3d()
        self.size = size
        self.linear = linear
        self.device = device
        if self.linear:
            self.linear_grid = build_affine_grid(self.size, self.device)

    def forward(self, source, target):
        x = self._encoder(source, target)
        grad = self.def_decoder(x)
        d1 = gradient_grid_3d(grad)
        if not self.linear:
            t = self.transformer(x, d1)
            return t, grad
        else:
            theta = self.a_decoder(x)
            d2 = affine_grid_3d(theta, self.linear_grid, self.size)
            t = self.transformer(x, d1, d2)
            return t, grad, theta
