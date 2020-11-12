import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import utils_imagenet as utils
from models import ResNet18_Imagenet, ResNet34_Imagenet, ResNet50_Imagenet, ResNet101_Imagenet, ResNet152_Imagenet
from opt.gopt_SGD import gSGD_SGD
from arguments import get_args

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))


best_acc1 = 0


def main():
    args = get_args()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        raise NotImplementedError
        print("=> using pre-trained model '{}'".format(args.model))
        model = models.__dict__[args.model](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.model))
        if args.model == "resnet18":
            model = ResNet18_Imagenet()
        elif args.model == "resnet34":
            model = ResNet34_Imagenet()
        elif args.model == "resnet50":
            model = ResNet50_Imagenet()
        elif args.model == "resnet101":
            model = ResNet101_Imagenet()
        elif args.model == "resnet152":
            model = ResNet152_Imagenet()
        else:
            raise NotImplementedError
        #
        # else:
        #     model = models.__dict__[args.model]()

    sys.stdout.flush()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.model.startswith('alexnet') or args.model.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.wd)
    elif args.optimizer == 'gsgd':
        optimizer = gSGD_SGD(models=model,
                             lr = args.lr,
                             args = args,
                             momentum=args.momentum,
                             weight_decay=args.wd,
                             bn=True)
    else:
        raise NotImplementedError("invalid opt for training imagenet")



    # if args.opt == "ec":
    #     # some info of gsgd
    #     if args.model == 'resnet50':
    #         model_info = {
    #             'class': "resnet_bottleneck",
    #             'resnet_blocks': [3, 4, 6, 3]
    #         }
    #     elif args.model == "resnet101":
    #         model_info = {
    #             'class': "resnet_bottleneck",
    #             'resnet_blocks': [3, 4, 23, 3]
    #         }
    #     elif args.model == "resnet152":
    #         model_info = {
    #             'class': "resnet_bottleneck",
    #             'resnet_blocks': [3, 8, 36, 3]
    #         }
    #     optim_info = {
    #         'class': 'sgd'
    #     }
    #     optimizer = gSGD(models=model, lr=args.lr, model_info=model_info, optim_info=optim_info,
    #                      momentum=args.momentum, weight_decay=args.wd, nesterov=args.nag == 'true', args=args)
    #     optimizer.step_choice = args.step
    # elif args.opt == "puresgd":


    # optionally resume from a checkpoint
    if args.resume:
        raise NotImplementedError
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    print('Loading data from zip file')
    if not args.debug:
        traindir = os.path.join(args.data_dir, 'train.zip')
        validdir = os.path.join(args.data_dir, 'validation.zip')
    else:
        traindir = os.path.join(args.data_dir, 'train_small.zip')
        validdir = os.path.join(args.data_dir, 'train_small.zip')
    print('Loading data into memory')
    sys.stdout.flush()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_data = utils.InMemoryZipDataset(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), num_workers=32)
    valid_data = utils.InMemoryZipDataset(validdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), num_workers=32)
    print('Found {} in training data'.format(len(train_data)))
    print('Found {} in validation data'.format(len(valid_data)))
    sys.stdout.flush()
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)

    val_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # decay_step = [30, 60, 90]
        decay_step = []
        # for one_step in args.decay_step :
        #     decay_step.append(int(one_step))
        if 'gsgd' not in args.optimizer:
            adjust_learning_rate(optimizer, epoch, args, args.decay_step)
        else:
            optimizer.adjust_learning_rate_multistep(  epoch,   args.decay_step,  args.decay_gamma  )

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            checkpoint_dic = {
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
            }
            if 'gsgd' not in args.optimizer:
                checkpoint_dic['optimizer'] = optimizer.state_dict()
            if (args.storage_frequency >0 ) and ((epoch % args.storage_frequency == 0) or (epoch == args.epochs -1 )):
                save_checkpoint(checkpoint_dic, is_best, check_point_name=args.check_point_name)
            else:
                save_checkpoint(checkpoint_dic, is_best, check_point_name=args.check_point_name)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        if  "gsgd" in args.optimizer:
            optimizer.epoch_num = epoch
            optimizer.iter_num = i

        # compute gradient and do SGD/G-SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        sys.stdout.flush()

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', check_point_name="", storage_check_point=""):
    if check_point_name == "" and storage_check_point != "":
        check_point_filename = "_".join([storage_check_point, filename])
        best_filename = "_".join([storage_check_point, 'model_best.pth.tar'])
        torch.save(state, check_point_filename)
        if is_best:
            shutil.copyfile(check_point_filename, best_filename)
    elif check_point_name != "" and storage_check_point == "":
        check_point_filename = "_".join([check_point_name, filename])
        best_filename = "_".join([check_point_name, 'model_best.pth.tar'])
        torch.save(state, check_point_filename)
        if is_best:
            shutil.copyfile(check_point_filename, best_filename)
    elif check_point_name != "" and storage_check_point != "":
        check_point_filename = "_".join([check_point_name, filename])
        best_filename = "_".join([check_point_name, 'model_best.pth.tar'])
        torch.save(state, check_point_filename)
        if is_best:
            shutil.copyfile(check_point_filename, best_filename)
        shutil.copyfile(check_point_filename, "_".join([storage_check_point, filename]))
        shutil.copyfile(best_filename, "_".join([storage_check_point, 'model_best.pth.tar']))
    else:
        raise ValueError("no save path assigned!")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args, decay_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    # lr = args.lr * (args.decay_gamma_imagenet ** (epoch // 30))
    if (epoch + 1) in decay_step:
        lr = args.lr * (args.decay_gamma ** (decay_step.index(epoch + 1) + 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
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


if __name__ == '__main__':
    main()
