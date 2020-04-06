import torch
import torch.nn as nn
import torch.optim as optim
from models.autoencoder import *
from models.register_3d import Register3d

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


class Trainer:
    def __init__(self, args, data, device=device):
        self._args = args
        self._data = data
        self._encoder = encoder
        self._model = Register3d(
            trainloader[0].size, device=device, linear=affine_transform).to(device)
        self._loss = nn.MSELoss()
        self._build_dir()
        self._build_optimizer()

    def build_dir(self):
        self._dir = os.path.join(
            os.getcwd(), 'temp', datetime.now().strftime('%m-%d_%H-%M-%S'))

    def build_optimizer(self):
        if self._args.optimizer == 'adam':
            optimizer_class = optim.Adam
        elif self._args.optimizer == 'sgd':
            optimizer_class = optim.SGD
        self._optimizer = optimizer_class(self._model.parameters(), lr=1e-3)

    def net_step(self, source, target):
        self._optimizer.zero_grad()
        self._model.train()
        if affine_transform:
            output, deform_grads, theta = self._model(source, target)
        else:
            output, deform_grads = self._model(source, target)
        loss = self._loss(output, target)
        if affine_transform:
            loss += alpha * torch.sum(torch.abs(theta - torch.eye(3, 4)))
        loss += beta * torch.sum(
            torch.abs(deform_grads - torch.ones_like(deform_grads)))
        loss.backward()
        self._optimizer.step()

    def trainer():
        self._epoch += 1
        for source, target in data['train']:
            source, target = source.to(dev), target.to(dev)
            self.net_step()
        epoch_dir = self._dir + "/epoch_{}".format(self._epoch)
        os.makedirs(epoch_dir)
        ckp_dir = epoch_dir + "/checkpoint.pt"
        torch.save(self._model.state_dict(), ckp_dir)

    def net_eval():
        model.eval()

        out, val_dgrads, val_theta = self._model(source, target)
        val_loss = loss_func(out, target)
        val_loss += alpha * \
            torch.sum(torch.abs(val_theta - torch.eye(3, 4)))
        + beta * torch.sum(torch.abs(val_dgrads -
                                     torch.ones_like(val_dgrads)))
        total_val_loss += val_loss.item()
        total_val_loss /= len(data['test'])
        print("Validation loss: {}".format(total_val_loss))


def evaluator():
    total_val_loss = 0
    with torch.no_grad():
        for source, target in data['val']:
            self.net_eval()
