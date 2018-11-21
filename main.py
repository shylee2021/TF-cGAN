import argparse

import tensorflow as tf

from dataset import get_mnist_dataset
from model import CGAN
from utils import print_with_time


def train(args):
    """ train model """
    batch_size = args.batch_size
    epochs = args.epochs
    base_lr = args.lr

    cgan = CGAN(args.name)
    train_dataset, _ = get_mnist_dataset(batch_size)

    with tf.Session() as sess:
        try:
            cgan.train(sess, train_dataset, base_lr=base_lr, epochs=epochs, save_period=10)
        except KeyboardInterrupt:
            print_with_time('Interrupted by user.')
        else:
            print_with_time('Training finished.')
        finally:
            print_with_time('Saving servable..')
            cgan.export(sess, export_dir='export/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='name of model', default='cGAN')
    parser.add_argument('-b', '--batch_size', help='batch size', default=128, type=int)
    parser.add_argument('-e', '--epochs', help='the number of epochs', default=300, type=int)
    parser.add_argument('-l', '--lr', help='base learning rate', default=3e-8, type=float)
    parsed_args = parser.parse_args()

    train(parsed_args)
