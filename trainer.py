import torch
import torch.nn as nn
import torch.optim as optim
from encoder import Encoder
from decoder import LinDecoder, DefDecoder
from sampling import sampler

encoder = Encoder(in_channels)
lin_decoder = LinDecoder(in_channels)
def_decoder = DefDecoder(in_channels)


def transformer_loss(out_lin, out_def):
    loss = torch.dist(
        self._data[1], self._sampler(self._data[0], G), p=2)
    affine_loss = self._alpha *
    torch.dist(out_lin, id_transformation_matrix, p=1)
    def_loss = self._beta *
    torch.dist(out_lin + out_def / 2, def_deformation, p=1)  # ??
    loss += affine_loss
    loss += def_loss
    return loss


class Trainer:
    def __init__(self, args, data, encoder, lin_decoder, def_decoder, sampler):
        self._args = args
        self._data = data
        self._encoder = encoder
        self._lin_decoder = lin_decoder
        self._def_decoder = def_decoder
        self._sampler = sampler
        self._epoch = 0
        self._build_dir()
        self._build_optimizer()
        self._build_loss()

    def _build_dir(self):
        self._dir = os.path.join(
            os.getcwd(), 'temp', datetime.now().strftime('%m-%d_%H-%M-%S'))

    def _build_optimizer(self):
        if self._args.optimizer == 'adam':
            optimizer_class = optim.Adam
        elif self._args.optimizer == 'sgd':
            optimizer_class = optim.SGD
        self._optimizer = optimizer_class(list(self._encoder.parameters()) +
                                          list(self._lin_decoder.parameters()) +
                                          list(self._def_decoder.parameters()),
                                          lr=self._args.learning_rate,
                                          weight_decay=self._args.weight_decay)

    def _net_step(self):
        self._encoder.train()
        self._lin_decoder.train()
        self._def_decoder.train()

        x = self._encoder(data)
        out_lin = self._lin_decoder(x)
        out_def = self._def_decoder(x)

        self._optimizer.zero_grad()

        loss = transformer_loss(out_lin, out_def)
        loss.backward()

        self._optimizer.step()

    def trainer():
        self._epoch += 1
        for batch_id, (S, R) in enumerate(self._data):
            print(batch_id)
            self._net_step()
        epoch_dir = self._dir + "/epoch_{}".format(self._epoch)
        os.makedirs(epoch_dir)
        ckp_dir_enc = epoch_dir + "/checkpoint_enc.pt"
        ckp_lin_enc = epoch_dir + "/checkpoint_lin.pt"
        ckp_def _enc = epoch_dir + "/checkpoint_def.pt"
        torch.save(self._encoder.state_dict(), ckp_dir_enc)
        torch.save(self._lin_decoder.state_dict(), ckp_dir_enc)
        torch.save(self._def_decoder.state_dict(), ckp_dir_enc)
