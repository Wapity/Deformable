import torch
import argparse

from encoder import Encoder
from decoder import Decoder, LinDecoder, DefDecoder
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',
                    help='batch size',
                    type=int,
                    default=10)
parser.add_argument('--optimizer',
                    help='optimizer type : ',
                    type=str,
                    default='adam')
parser.add_argument('--learning_rate',
                    help='learning rate for optimizer',
                    type=float,
                    default=1e-3)
parser.add_argument('--learning_rate_decay',
                    help='learning rate decay for optimizer',
                    type=float,
                    default=1.)
parser.add_argument('--weight_decay',
                    help='weight decay for optimizer',
                    type=float,
                    default=0.)

parser.add_argument('--alpha',
                    help='regularization weight for affine loss',
                    type=float,
                    default=1e-6)
parser.add_argument('--beta',
                    help='regularization weight for deformable loss',
                    type=float,
                    default=1e-6)


trainer = Trainer(args, Encoder(), LinDecoder(), DefDecoder())
trainer.train()
