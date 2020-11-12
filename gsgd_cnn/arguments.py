import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='gsgd')


    ### dataset ######
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--dataset' , default='cifar')
    parser.add_argument('--data_dir', type=str, default='./')

    ### training ####
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='start epochs to train (default: 10)')


    ### optim #####
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0., metavar='M',
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--alpha_rmsprop', type=float, default=0.99, metavar='M',
                        help='RMSprop alpha (default: 0.99)')
    parser.add_argument('--eps_rmsprop', type=float, default=1e-8, metavar='M',
                        help='RMSprop eps (default: 1e-8)')
    parser.add_argument('--wd', type=float, default=0., metavar='W',
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--decay', choices=['poly', 'linear', 'multistep', 'exp', 'none'], default='multistep')

    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='decay gamma')
    parser.add_argument('--decay_step', default=[80,120], type=int, nargs='+',
                        help='decay steps ')
    parser.add_argument('--pow', type=float, default=0.5,
                        help='poly decay power')

    parser.add_argument('--optimizer', choices=['sgd', 'rmsprop', 'adam', 'gsgd', 'gsgd_rmsprop', 'gsgd_adam'],
                        default='gsgd')


    ## init ####
    parser.add_argument('--red_init', type=float, default=1.0, metavar='W',
                        help='red edge init (default: 1.0)')
    parser.add_argument('--first_red_init', type=float, default=1.0, metavar='W',
                        help='red edge init (default: 1.0)')
    parser.add_argument('--blue_init', type=float, default=1.0, metavar='W',
                        help='blue edge init (default: 1.0)')
    parser.add_argument('--bn_init', type=float, default=1.0)


    ## others ####
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: random)')
    parser.add_argument('--seedm', type=int, default=1, metavar='S',
                        help='random seed multiple')

    parser.add_argument('--gpu', default=None, type=int)


    ## log and save ###
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log', type=str, default='')
    parser.add_argument('-p', '--print_freq', default=2000, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--storage_frequency', default=0, type=int,
                        help='frequency save to storage.')

    #### model ####

    parser.add_argument('--model',  default='plain34')


    ## ec ##
    parser.add_argument('--ec_layer_type', default=0, type=int)
    parser.add_argument('--ec_opt', default='ec', type=str)
    parser.add_argument('--convid', choices=[None,'resnet34_1' , 'resnet34_2', 'resnet110_1', 'resnet110_2'], default=None)
    parser.add_argument('--inner_opt', default='sgd', type=str)
    parser.add_argument('--rmsprop_method', default='weight', type=str)
    parser.add_argument('--adam_method', default='weight', type=str)

    parser.add_argument('--mom_method', default='weight', type=str)
    # parser.add_argument('--momentum_method', default='weight', type=str)
    parser.add_argument('--check_point_name', default='0', type=str,
                        help='name of the checkpoint')


    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--debug', action='store_true',
                        help='load small datasets to debug.')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')


    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # args.vis = not args.no_vis

    return args
