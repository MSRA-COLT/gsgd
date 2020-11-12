import torch
import numpy as np
from torch import nn
from model.mnist import MnistNet



from model.pathnet import  PathNet
from model.models import *# SimpleNet_den, SimpleNet_den_mnist
# from opt.gopt_general_mlp_unequal import  gSGD
from opt.gopt_general_mlp import gSGD
# from gopt import gSGD as gSGD_old

import matplotlib
import copy
import time
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets, transforms
from data_loader.dataset import simulation_data, mnist, cifar10
from mpl_toolkits.mplot3d import Axes3D
import time
from matplotlib import animation
from IPython.display import HTML
# from models import SimpleNet_mnist
# matplotlib.rcParams['animation.embed_limit'] = 2 ** 128

from itertools import zip_longest
import os
from arguments import get_args
import copy

args = get_args()

args.device = torch.device("cuda:0" if args.cuda else "cpu")

if args.seed  == -1000:
    args.seed = int(time.time() * 1000)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.deterministic=True
    torch.cuda.manual_seed_all(args.seed)

# args.device = torch.device("cuda" if args.cuda else "cpu")

print(args)


criterion = nn.CrossEntropyLoss().to(args.device)


if args.model=='mnistmlp':
    # Model = MnistNet
    Model = PathNet(input_size= 28*28)
elif args.model == 'cifarmlp':
    Model = PathNet(input_size=32*32*3 )
elif args.model == 'simple_den':
    Model = SimpleNet_den((3,8,2,8,2,8,2,8,3),input_size=10)
elif args.model == 'simple_den_mnist':
    Model = SimpleNet_den_mnist()
elif args.model == 'simple_den_mnist_2':
    Model = SimpleNet_den_mnist_2()
elif args.model == 'simple_den_mnist_3':
    Model = SimpleNet_den_mnist_3()


model = Model.to(args.device)

# for ii in net_gsgd.parameters():
    # print(ii.data.device, args.device)
    # assert ii.data.device == torch.device('cuda')
optimizer = gSGD(model, lr= args.lr, args=args)

#
#
# print(list(model.parameters())[3],list(model_true.parameters())[3])



rec_sgd = []


if args.data =='mnist':
    train_loader, test_loader = mnist(args.batch_size,1000)
elif args.data == 'cifar':
    train_loader, test_loader = cifar10(args.batch_size, 1000)
elif args.data == 'simulation':
    model_true = SimpleNet_den((3, 8, 2, 8, 2, 8, 2, 8, 3), input_size=10)
    train_loader, test_loader = simulation_data(n_data_train=100,
                                                n_data_test=50,
                                                n_feature=10,
                                                noise_y=0.5,
                                                noise_model= 0.5,
                                                model=model_true,
                                                train_batch_size=64,
                                                test_batch_size=256)




torch.backends.cudnn.benchmarks = True


all_data_number_train = len(train_loader.dataset)
all_batch_number_train = len(train_loader)
all_data_number_test = len(test_loader.dataset)
all_batch_number_test = len(test_loader)

def train(epoch):
    model.train()
    loss_interval = 0
    train_loss = 0
    correct = 0

    # for current_batch, (data, label) in enumerate(tqdm.tqdm(training_data, ncols = 0)):
    # scheduler.step()
    for batch_idx, (data, label) in enumerate( train_loader):

        data = data.to(args.device)

        label =  label.to(args.device)


        output = model(data)

        loss = criterion(output, label)


        optimizer.zero_grad()
        loss.backward()
        # for ii in model.parameters():
        #     print(ii.grad.data)

        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        loss_interval += loss.item()
        train_loss += loss.item()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), all_data_number_train,
                       100. * batch_idx / all_batch_number_train, loss_interval / args.log_interval))

            loss_interval = 0



    train_loss /= batch_idx + 1

    print('\n[Training] Epoch: {} Average loss: {}, Accuracy: {}/{} ({})\n'.format(
        epoch, train_loss, correct, all_data_number_train,
        float(correct) / all_data_number_train))



def test(epoch):
    model.eval()
    test_loss =0
    correct = 0
    total = 0

    # print(epoch, ' ## Testing  ...')

    for data, label in test_loader:
        total += label.size(0)
        data = data.to(args.device)
        label = label.to(args.device)



        output = model(data)
        loss = criterion(output, label)
        test_loss += loss.item() * label.shape[0]
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
    test_loss /= all_data_number_test
    print('\n[Vaidation] Epoch: {} Average loss: {}, Accuracy: {}/{} ({})\n'.format(
        epoch, test_loss, correct, all_data_number_test,
        float(correct) / all_data_number_test))


        # _, predicted = torch.max(output.data, 1)

        # correct += (predicted == label.data).sum().item()

    # print(epoch, ' ## Testing accurracy ...      ' , str(100 * correct / total) + '%')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    t1 = time.time()
    for epoch in range(1, args.epochs + 1):
        optimizer.lr_decay(decay=args.decay , epoch= epoch , max_epoch=args.epochs, decay_state={'power':args.power, 'step': args.decay_step, 'gamma': args.gamma})
        train(epoch)
        test(epoch)
    t2 = time.time()
    # print(t2-t1)

