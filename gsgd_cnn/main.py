from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# from torch.autograd import Variable
# from torch.optim import Optimizer
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np
import time


from models import *
# from opt.gopt_RMSprop import gSGD_RMSprop
# from opt.gopt_Adam import gSGD_Adam
# from opt.gopt import    gSGD as gSGD_naive
# from torch.optim import SGD, RMSprop, Adam
# from torch.optim.lr_scheduler import MultiStepLR


from opt.gopt_SGD import gSGD_SGD
from arguments import get_args


from arguments import get_args

from config import user_def_conv_idx




args = get_args()
if args.seed  == -1000:
    args.seed = int(time.time() * 1000)
args.seed = args.seed * args.seedm
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.deterministic=True
    torch.cuda.manual_seed_all(args.seed)


args.device = torch.device("cuda:0" if args.cuda else "cpu")
print(args)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

if args.dataset == 'mnist':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

elif args.dataset == 'fashion':
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data/fashion', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data/fashion', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
elif args.dataset == 'cifar':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=1)

elif args.dataset == 'cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    trainset = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    testset = datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)
elif args.dataset == 'imagenet':
    raise  NotImplementedError('imagenet')
    # Baseline: https://github.com/tensorflow/models/tree/master/research/slim
    # imagenet_datadir = r'/home/shuzhe/dataset/raw-data/'
    # traindir = os.path.join(imagenet_datadir, 'train')
    # valdir = os.path.join(imagenet_datadir, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    #
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(traindir,
    #                          transforms.Compose([
    #                              transforms.RandomSizedCrop(224),
    #                              transforms.RandomHorizontalFlip(),
    #                              transforms.ToTensor(),
    #                              normalize,
    #                          ])),
    #     batch_size=args.batch_size * torch.cuda.device_count(), shuffle=True,
    #     num_workers=8, pin_memory=True)
    #
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size * torch.cuda.device_count(), shuffle=False,
    #     num_workers=8, pin_memory=True)

if args.model == 'mnist':
    model = MNIST_Net()
elif args.model == 'cifar':
    model = CIFAR_Net()
elif args.model == 'cifarbn':
    model = CIFAR_BNNet()
elif args.model == 'plain110':
    model = PlainNet110()
elif args.model == 'plain34':
    model = PlainNet34()
elif args.model == 'plain10':
    model = PlainNet10()
elif args.model == 'plain14':
    model = PlainNet14()
elif args.model == 'ac10':
    model = ACNet10()
elif args.model == '2ac10':
    model = ACNet2_10()
elif args.model == '2ac34':
    model = ACNet2_34()
elif args.model == 'resnet50':
    model = ResNet50()
elif args.model == 'resnet34':
    model = ResNet34()
elif args.model == 'resnet110':
    model = ResNet110()
elif args.model == 'resnet34w':
    model = ResNet_w34()
elif args.model == 'resnet110w':
    model = ResNet_w110()
elif args.model == 'vgg16':
    import torchvision.models as models
    model = models.__dict__['vgg16']()
elif args.model == 'plainb10':
    model = PlainNetB10()
elif args.model == 'plainb34':
    model = PlainNetB34()
elif args.model == 'resnet50i':
    import torchvision.models as models
    model = models.__dict__['resnet50']()
elif args.model == 'rnnm':
    model = RNN_Mnist()
elif args.model == 'plain34c100':
    model = PlainNet34_C100()
elif args.model == 'plain110c100':
    model = PlainNet110_C100()
elif args.model == 'resnet34c100':
    model = ResNet34_C100()
elif args.model == 'resnet110c100':
    model = ResNet110_C100()
elif args.model == 'resnet34c100w':
    model = ResNet_w34_C100()
# elif args.model == 'resnet34c100w_ud':
#     model = ResNet_w34_C100_userdef()
elif args.model == 'resnet110c100w':
    model = ResNet_w110_C100()
elif args.model == 'resnet18':
    model = ResNet18()
elif args.model == 'resnet18c100':
    model = ResNet18_C100()
elif args.model == 'resnet18w':
    model = ResNet_w18()
elif args.model == 'resnet18c100w':
    model = ResNet_w18_C100()


elif args.model == 'densenet21':
    model = DenseNet21()
elif args.model == 'densenet37':
    model = DenseNet37()
elif args.model == 'densenet45':
    model = DenseNet45()
elif args.model == 'densenet85':
    model = DenseNet85()
elif args.model == 'densenet121':
    model = DenseNet121()


elif args.model == 'densenet21c100':
    model = DenseNet121C100()
elif args.model == 'densenet37c100':
    model = DenseNet37C100()
elif args.model == 'densenet45c100':
    model = DenseNet45C100()
elif args.model == 'densenet85c100':
    model = DenseNet85C100()
elif args.model == 'densenet121c100':
    model = DenseNet121C100()


#
#
# elif args.model == 'densenet21my':
#     model = DenseNet21_my()
# elif args.model == 'densenet37my':
#     model = DenseNet37_my()
# elif args.model == 'densenet45my':
#     model = DenseNet45_my()
# elif args.model == 'densenet85my':
#     model = DenseNet85_my()
# elif args.model == 'densenet121my':
#     model = DenseNet121_my()
#
#
# elif args.model == 'densenet21c100my':
#     model = DenseNet121C100_my()()
# elif args.model == 'densenet37c100my':
#     model = DenseNet37C100_my()
# elif args.model == 'densenet45c100my':
#     model = DenseNet45C100_my()
# elif args.model == 'densenet85c100my':
#     model = DenseNet85C100_my()
# elif args.model == 'densenet121c100my':
#     model = DenseNet121C100_my()


elif args.model == 'densenet40':
    model = DenseNet40()
elif args.model == 'densenet100':
    model = DenseNet100()
elif args.model == 'densenet40c100':
    model = DenseNet40C100()
elif args.model == 'densenet100c100':
    model = DenseNet40C100()



# elif args.model == 'path':
#     if args.dataset == 'cifar':
#         model = PathNet(input_size=32 * 32 * 3, hidden_size=args.hsize, layer=args.pathlayer)
#     elif args.dataset == 'mnist' or args.dataset == 'fashion':
#         model = PathNet(input_size=28 * 28, hidden_size=args.hsize, layer=args.pathlayer)
#     else:
#         raise Exception('only cifar and mnist')
# elif args.model == 'vggnet':
#     model = VGG16()
#


#
# if args.dataset == 'imagenet':
#     model = torch.nn.parallel.DataParallel(model)


if args.model == 'resnet34' or args.model == 'resnet34c100':
    model_info ={
        'class' : "resnet",
        'resnet_blocks' :  [3,4,6,3]
    }
elif args.model == 'resnet18'  or args.model == 'resnet18c100':
    model_info ={
        'class' : "resnet",
        'resnet_blocks' :  [2,2,2,2]
    }
elif args.model == 'resnet110'  or args.model == 'resnet110c100':
    model_info ={
        'class' : "resnet",
        'resnet_blocks' :  [3,8,40,3]
    }

elif args.model == 'resnet18w' or args.model == 'resnet18c100w' :
    model_info ={
        'class' : "resnet_w",
        'resnet_blocks' :  [2,2,2,2]
    }
elif args.model == 'resnet34w' or args.model == 'resnet34c100w' or args.model == 'resnet34w_ud' or args.model == 'resnet34c100w_ud':
    model_info ={
        'class' : "resnet_w",
        'resnet_blocks' :  [3,4,6,3]
    }
elif args.model == 'resnet110w'  or args.model == 'resnet110c100w' or args.model == 'resnet110w_ud' or args.model == 'resnet110c100w_ud':
    model_info ={
        'class' : "resnet_w",
        'resnet_blocks' :  [3,8,40,3]
    }

elif args.model == 'plain34' or args.model == 'plain34c100':
    model_info ={
        'class' : "plain",
        'plainnet_blocks' :  [6,8,12,6]
    }
elif args.model == 'plain110' or args.model == 'plain110c100':
    model_info ={
        'class' : "plain",
        'plainnet_blocks' :  [6,16,80,6]
    }

elif 'densenet' in args.model   :
    if 'my' in args.model:
        model_info ={
            'class' : "densenet_my",
            'densenet_blocks' :  model.nblocks
        }
    else:
        model_info ={
            'class' : "densenet",
            'densenet_blocks' :  model.nblocks
        }




optim_info={
    'class' : 'sgd'
}

model = model.to(args.device)


if args.convid is not None:
    model.conv_idx = user_def_conv_idx[args.convid]

# if args.opt == 'gsgd':

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.wd)
    # scheduler = MultiStepLR(optimizer, milestones=[150,225], gamma=0.1)
elif args.optimizer == 'gsgd':
    optimizer = gSGD_SGD(models=model,
                         lr = args.lr,
                         args = args,
                         momentum=args.momentum,
                         weight_decay=args.wd,
                         bn=True)


    # optimizer = gSGD_SGD(models=model, lr=args.lr,
    #                  momentum=args.momentum, weight_decay=args.wd, args=args)

# elif args.opt == 'gsgd_rmsprop':
#     optimizer = gSGD_RMSprop(models=model, lr=args.lr, args=args,
#                       alpha=args.alpha_rmsprop, eps=args.eps_rmsprop, bn=True)
#
# elif args.opt == 'gsgd_adam':
#     optimizer = gSGD_Adam(models=model, lr=args.lr, args=args,
#                         bn=True)

# elif args.opt == 'sgd':
#     optimizer = SGD(model.parameters(), lr=args.lr )


# elif args.opt == 'rmsprop':
#     optimizer = RMSprop(model.parameters(), lr=args.lr,alpha=args.alpha_rmsprop, eps=args.eps_rmsprop )
#     scheduler = MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
# elif args.opt == 'adam':
#     optimizer = Adam(model.parameters(), lr=args.lr )
#     scheduler = MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

# elif args.opt == 'rmsprop':
#     optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

print(optimizer)



# optimizer.step_choice = args.step


criterion = nn.CrossEntropyLoss().to(args.device)
start_epoch = 0


cudnn.benchmark = True


def train(epoch):
    model.train()
    loss_interval = 0
    train_loss = 0
    correct = 0
    correct_5 = 0
    time_0 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(time.time() - time_0)
        # time_0 =time.time()
        data = data.to(args.device)
        target = target.to(args.device)
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)
        if args.dataset == 'imagenet':
            top1, top5 = accuracy(output.data, target.data, topk=(1, 5))
            correct += top1[0]
            correct_5 += top5[0]
        else:
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        loss.backward()
        # print(batch_idx, loss.item())
        # print(optimizer.print_info())
        # print(batch_idx, ' ############################################ ')
        optimizer.step()
        loss_interval += loss.item()
        train_loss += loss.item()


        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss_interval / args.log_interval))

            loss_interval = 0
        # break

    train_loss /= batch_idx + 1
    if args.dataset == 'imagenet':
        print('\n[Training] Epoch: {} Average loss: {}, Top1 Accuracy: {}/{} ({}), Top5 Accuracy: {}/{} ({})\n'.format(
            epoch, train_loss, correct, len(train_loader.dataset),
            correct / len(train_loader.dataset), correct_5, len(train_loader.dataset),
            float(correct_5) / len(train_loader.dataset)))

    else:
        print('\n[Training] Epoch: {} Time: {:.2f}  Average loss: {}, Accuracy: {}/{} ({})\n'.format(
            epoch, time.time()-time_0, train_loss, correct, len(train_loader.dataset),
            float(correct) / len(train_loader.dataset)))

    if args.storage_frequency> 0 and epoch % args.storage_frequency == 0:
        torch.save([optimizer,model,], "{}-{}.pt".format( args.check_point_name, epoch,) )



def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    correct_5 = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if args.model == 'rnnm' and args.dataset == 'mnist':
            data, target = Variable(data.view(-1, 28, 28), volatile=True), Variable(target)
        else:
            pass
            # data, target = Variable(data, volatile=True), Variable(target)

        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.data * target.data.shape[0]  # sum up batch loss
        if args.dataset == 'imagenet':
            top1, top5 = accuracy(output.data, target.data, topk=(1, 5))
            correct += top1[0]
            correct_5 += top5[0]
        else:
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)

    if args.dataset == 'imagenet':
        print('\n[Vaidation] Epoch: {} Average loss: {}, Top1 Accuracy: {}/{} ({}), Top5 Accuracy: {}/{} ({})\n'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
            correct / len(test_loader.dataset), correct_5, len(test_loader.dataset),
            float(correct_5) / len(test_loader.dataset)))

    else:
        print('\n[Vaidation] Epoch: {} Average loss: {}, Accuracy: {}/{} ({})\n'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
            float(correct) / len(test_loader.dataset)))



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

def adjust_learning_rate(optimizer, epoch, args, decay_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    # lr = args.lr * (args.decay_gamma_imagenet ** (epoch // 30))
    if (epoch + 1) in decay_step:
        lr = args.lr * (args.decay_gamma ** (decay_step.index(epoch + 1) + 1))
        print('sgd lr decay to :', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__ == "__main__":

    # print(11111111)
    for epoch in range(start_epoch, args.epochs  ):
        # if args.dataset == 'imagenet':
        #     if args.model == 'resnet50i':
        #         decay_step = [30, 60, 80]
        #     elif args.model == 'vgg16':
        #         decay_step = [20, 40, 60, 70]
        # else:
        #     # decay_step = [80, 120]
        #     decay_step = [80,120]
        # if args.opt == 'sgd' or args.opt =='rmsprop' or args.opt =='adam':
        #     scheduler.step()
        # else:
        #     optimizer.lr_decay(args.decay, epoch - 1, args.epochs + 1, {'power': args.pow, 'step': decay_step, 'gamma': 0.1})
        #
        if 'gsgd' not in args.optimizer:
            adjust_learning_rate(optimizer, epoch, args, args.decay_step)
        else:
            optimizer.adjust_learning_rate_multistep(  epoch,   args.decay_step,  args.decay_gamma  )

        train(epoch)
        test(epoch)
