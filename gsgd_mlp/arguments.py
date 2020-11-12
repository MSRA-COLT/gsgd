import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='mlp argument')
    parser.add_argument('--data', type=str, choices=['mnist', 'cifar', 'simulation'], default='mnist',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str,  choices=['mnistmlp',  'cifarmlp',  'simulationmlp', 'simple_den', 'simple_den_mnist', 'simple_den_cifar', 'simple_den_mnist_2', 'simple_den_cifar_2', 'simple_den_mnist_3', 'simple_den_cifar_3'],
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    # parser.add_argument('--input_size', type=int, default=784,
    #                     help='number of input feature')
    # parser.add_argument('--num_hidden', type=int, default=100,
    #                     help='number of hidden units per layer')
    # parser.add_argument('--num_class', type=int, default=10,
    #                     help='number of output class')
    # parser.add_argument('--num_layer', type=int, default=1,
    #                     help='number of layers')
    # parser.add_argument('--clip', type=float, default= 1,
    #                     help='gradient_clipping_value')
    parser.add_argument('--lr', type=float, default=1,
                        help='initial learning rate')
    parser.add_argument('--opt', choices=['bp', 'ec', 'path'], default='ec')
    parser.add_argument('--ec_layer_type', type=int, default=1,
                        help='ec layer type')
    # parser.add_argument('--momentum', type=float, default=0., metavar='M',
    #                     help='SGD momentum (default: 0.0)')

    parser.add_argument('--epochs', type=int, default=1,
                        help='upper epoch limit')
    parser.add_argument('--decay', choices=['poly', 'linear', 'multistep', 'exp', 'none'], default='multistep')
    parser.add_argument('--power', type=float, default=0.01,
                        help='poly decay power')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multistep decay gamma')
    parser.add_argument('--decay_step', nargs='*', type=int, default=[25,50],
                        help='decay step, default is [25,50]')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size')
    # parser.add_argument('--bptt', type=int, default=35,
    #                     help='sequence length')
    # parser.add_argument('--dropout', type=float, default=0.2,
    #                     help='dropout applied to layers (0 = no dropout)')
    # parser.add_argument('--tied', action='store_true',
    #                     help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    # parser.add_argument('--onnx-export', type=str, default='',
    #                     help='path to export the final model in onnx format')
    args = parser.parse_args()

    args.cuda =   args.cuda and torch.cuda.is_available()
    # args.vis = not args.no_vis

    return args
