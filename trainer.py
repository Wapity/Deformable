import torch
import torch.nn as nn
import torch.optim as optim
from encoder import Encoder
from decoder import LinDecoder, DefDecoder
from sampling import sampler

encoder = Encoder(in_channels)
lin_decoder = LinDecoder(in_channels)
def_decoder = DefDecoder(in_channels)


class Trainer:
    def __init__(self, args, data, encoder, lin_decoder, def_decoder, sampler):
        self._args = args
        self._data = data
        self._encoder = encoder
        self._lin_decoder = lin_decoder
        self._def_decoder = def_decoder
        self._sampler = sampler

    def _build_optimizer(self):
        if self._args.optimizer == 'adam':
            optimizer_class = optim.Adam
        elif self._args.optimizer == 'sgd':
            optimizer_class = optim.SGD

    def _build_loss(self):
        # ///////////////////////////////////////////
        def loss():
            loss = torch.dist(self._data[1], self._sampler(self._data[0]), p=2)
            affine_loss = self._alpha * \
                torch.dist(self._lin_decoder(self._data), id_deformation, p=1)
            def_loss = self._beta * \
                torch.dist(self._def_decoder(self._data), def_deformation, p=1)
            loss += affine_loss
            loss += def_loss
            return loss

    def _net_step(self):
        self._encoder.train()
        self._lindecoder.train()
        self._defdecoder.train()
        x = self._encoder(data)
        ###########

    def trainer():
        return None
