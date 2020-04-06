import torch
import argparse

from autoencoder import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',
                    help='batch size',
                    type=int,
                    default=1)
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
parser.add_argument('--train',
                    help='Train or Eval',
                    type=bool,
                    default=True)

args = parser.parse_args()
trainer = Trainer(args, Encoder(), LinDecoder(), DefDecoder())

if args.train:
    trainer.train()
else:
    trainer.eval()
