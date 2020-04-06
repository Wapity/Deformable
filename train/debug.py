import torch
import torch.nn as nn

from models.encoder import Encoder
from models.decoder import LinDecoder, DefDecoder


encoder = Encoder(2)
lin_decoder = LinDecoder(290)
def_decoder = DefDecoder(290)

x = [torch.normal(1, 0.5, size=(1, 1, 64, 192, 192)),
     torch.normal(1, 0.5, size=(1, 1, 64, 192, 192))]

out = encoder(*x)
print(out)
