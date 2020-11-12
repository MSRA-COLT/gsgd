import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import Optimizer
import torch.nn.parallel
import numpy as np
import scipy.sparse as sparse
import torchvision.models

import random
import math

# from utils import resnet_conv_id
# from models import *


class SGD(Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)
        self.ec_layer = None

    def cal_ec_layer(self, ec_layer):
        self.ec_layer = ec_layer




    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        # assert 1==2
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            # dampening = momentum
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data * lr
                if weight_decay != 0:
                    d_p.add_(weight_decay * lr, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).cuda()
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_((1 - dampening), d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-1, d_p)
        return loss

    def partial_bp_step(self, conv_idx, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            # dampening = momentum
            nesterov = group['nesterov']

            for layer_idx, p in enumerate(group['params']):
                isinconvidx = False
                if isinstance(conv_idx[0][0], list):
                    for layer in conv_idx:
                        for sub_layer in layer:
                            isinconvidx = isinconvidx or (layer_idx in sub_layer)
                else:
                    for layer in conv_idx:
                        isinconvidx = isinconvidx or (layer_idx in layer)

                if isinconvidx:
                    continue


                if p.grad is None:
                    continue
                d_p = p.grad.data * lr
                if weight_decay != 0:
                    d_p.add_(weight_decay * lr, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).cuda()
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_((1 - dampening), d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-1, d_p)
        return loss



    def unpacklist(self, a, b):
        if not isinstance(a, list):
            b.append(a)
        else:
            for ii in a:
                self.unpacklist(ii, b)

    def partial_bp_step_w(self, closure=None):
        if self.ec_layer is None:
            raise Exception('please call cal_ec_layer first')
        # print(self.ec_layer, '!!!!')
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            # dampening = momentum
            nesterov = group['nesterov']

            for layer_idx, p in enumerate(group['params']):
                isinconvidx = layer_idx in self.ec_layer

                if isinconvidx:

                    continue
                # print(layer_idx)
                #


                if p.grad is None:
                    continue
                d_p = p.grad.data * lr
                if weight_decay != 0:
                    d_p.add_(weight_decay * lr, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).cuda()
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_((1 - dampening), d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-1, d_p)
        return loss




'''
    def path_step(self, gammain, gammaout, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            # dampening = momentum
            nesterov = group['nesterov']

            for layer_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data * lr / (gammaout[layer_idx].mm(gammain[layer_idx].transpose(0, 1)))

                p.data.add_(-1, d_p)
        return loss

    def cntk_step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            # dampening = momentum
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).cuda()
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_((1 - momentum), d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-lr, d_p)
        return loss

    def fake_step(self, conv_idx, linear_idx, conv_mask, linear_mask, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            # dampening = momentum
            nesterov = group['nesterov']

            for layer_idx, p in enumerate(group['params']):
                isinconvidx = False

                if isinstance(conv_idx[0][0], list):
                    for idx, layer in enumerate(conv_idx):
                        for idx2, sub_layer in enumerate(layer):
                            isinconvidx = isinconvidx or (layer_idx in sub_layer)
                            if layer_idx in sub_layer:
                                mask = conv_mask[idx][idx2][sub_layer.index(layer_idx)]
                else:
                    for idx, layer in enumerate(conv_idx):
                        isinconvidx = isinconvidx or (layer_idx in layer)
                        if layer_idx in layer:
                            mask = conv_mask[idx][layer.index(layer_idx)]

                isinlinearidx = layer_idx in linear_idx
                if not isinconvidx and not isinlinearidx:
                    continue
                # print(type(p), type(p).__name__)
                if p.grad is None:
                    continue
                d_p = p.grad.data * lr
                if weight_decay != 0:
                    d_p.add_(weight_decay * lr, p.data * (1 - mask))
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).cuda()
                        # buf.mul_(momentum).add_(d_p)
                        buf = (buf * momentum + d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        # buf.mul_(momentum).add_((1 - dampening), d_p)
                        buf = (buf * momentum + (1 - dampening) * d_p)
                    if nesterov:
                        d_p = d_p + momentum * buf * (1 - mask)
                    # d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf * (1 - mask) + mask * d_p

                p.grad.data = d_p / lr

        return loss

    def fake_scale_step(self, conv_idx, linear_idx, conv_mask, linear_mask, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            # dampening = momentum
            nesterov = group['nesterov']

            for layer_idx, p in enumerate(group['params']):
                isinconvidx = False

                if isinstance(conv_idx[0][0], list):
                    for idx, layer in enumerate(conv_idx):
                        for idx2, sub_layer in enumerate(layer):
                            isinconvidx = isinconvidx or (layer_idx in sub_layer)
                            if layer_idx in sub_layer:
                                mask = conv_mask[idx][idx2][sub_layer.index(layer_idx)]
                else:
                    for idx, layer in enumerate(conv_idx):
                        isinconvidx = isinconvidx or (layer_idx in layer)
                        if layer_idx in layer:
                            mask = conv_mask[idx][layer.index(layer_idx)]

                isinlinearidx = layer_idx in linear_idx
                if not isinconvidx and not isinlinearidx:
                    continue
                # print(type(p), type(p).__name__)
                if p.grad is None:
                    continue
                w_norm = p.data.norm(2, 1).unsqueeze(1)
                d_p = p.grad.data * lr * w_norm
                if weight_decay != 0:
                    d_p.add_(weight_decay * lr, p.data * (1 - mask))
                if momentum != 0:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).cuda()
                        # buf.mul_(momentum).add_(d_p.mul(w_norm))
                        buf = (buf * momentum + d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        # buf.mul_(momentum).add_((1 - dampening), d_p.mul(w_norm))
                        buf = (buf * momentum + (1 - dampening) * d_p)
                    if nesterov:
                        # d_p = d_p.add(momentum, buf)
                        d_p = d_p + momentum * buf * (1 - mask)
                    else:
                        d_p = buf * (1 - mask) + d_p * mask

                p.grad.data = d_p / lr

        return loss

    def fake_mom_step(self, conv_idx, linear_idx, conv_mask, linear_mask, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            # dampening = momentum
            nesterov = group['nesterov']

            for layer_idx, p in enumerate(group['params']):
                isinconvidx = False

                if isinstance(conv_idx[0][0], list):
                    for idx, layer in enumerate(conv_idx):
                        for idx2, sub_layer in enumerate(layer):
                            isinconvidx = isinconvidx or (layer_idx in sub_layer)
                            if layer_idx in sub_layer:
                                mask = conv_mask[idx][idx2][sub_layer.index(layer_idx)]
                else:
                    for idx, layer in enumerate(conv_idx):
                        isinconvidx = isinconvidx or (layer_idx in layer)
                        if layer_idx in layer:
                            mask = conv_mask[idx][layer.index(layer_idx)]

                isinlinearidx = layer_idx in linear_idx
                if not isinconvidx and not isinlinearidx:
                    continue
                # print(type(p), type(p).__name__)
                if p.grad is None:
                    continue
                d_p = p.grad.data * lr
                # if weight_decay != 0:
                # 	d_p.add_(weight_decay * lr, p.data * (1-mask))
                if momentum != 0:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).cuda()
                        # buf.mul_(momentum).add_(d_p.mul(w_norm))
                        buf = (buf * momentum + d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        # buf.mul_(momentum).add_((1 - dampening), d_p.mul(w_norm))
                        buf = (buf * momentum + (1 - dampening) * d_p)
                    if nesterov:
                        # d_p = d_p.add(momentum, buf)
                        d_p = d_p + momentum * buf * (1 - mask)
                    else:
                        d_p = buf * (1 - mask) + mask * d_p

                p.grad.data = d_p

        return loss

    def fake_scale_mom_step(self, conv_idx, linear_idx, conv_mask, linear_mask, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            # dampening = momentum
            nesterov = group['nesterov']

            for layer_idx, p in enumerate(group['params']):
                isinconvidx = False

                if isinstance(conv_idx[0][0], list):
                    for idx, layer in enumerate(conv_idx):
                        for idx2, sub_layer in enumerate(layer):
                            isinconvidx = isinconvidx or (layer_idx in sub_layer)
                            if layer_idx in sub_layer:
                                mask = conv_mask[idx][idx2][sub_layer.index(layer_idx)]
                else:
                    for idx, layer in enumerate(conv_idx):
                        isinconvidx = isinconvidx or (layer_idx in layer)
                        if layer_idx in layer:
                            mask = conv_mask[idx][layer.index(layer_idx)]

                isinlinearidx = layer_idx in linear_idx
                if not isinconvidx and not isinlinearidx:
                    continue
                # print(type(p), type(p).__name__)
                if p.grad is None:
                    continue
                w_norm = p.data.norm(2, 1).unsqueeze(1)
                # print(w_norm)
                d_p = p.grad.data * w_norm * (lr * (1 - mask) + lr / (1 - momentum) * mask)
                # d_p = p.grad.data * w_norm * lr
                # if weight_decay != 0:
                # 	d_p.add_(weight_decay * lr, p.data * (1-mask))
                if momentum != 0:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).cuda()
                        # buf.mul_(momentum).add_(d_p.mul(w_norm))
                        buf = (buf * momentum + d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        # buf.mul_(momentum).add_((1 - dampening), d_p.mul(w_norm))
                        buf = (buf * momentum + (1 - dampening) * d_p)
                    if nesterov:
                        # d_p = d_p.add(momentum, buf)
                        d_p = d_p + momentum * buf * (1 - mask)
                    else:
                        d_p = buf * (1 - mask) + mask * d_p
                    # d_p = buf

                p.grad.data = d_p

        return loss

    def cntk_fake_scale_mom_step(self, conv_idx, linear_idx, conv_mask, linear_mask, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            # dampening = momentum
            nesterov = group['nesterov']

            for layer_idx, p in enumerate(group['params']):
                isinconvidx = False

                if isinstance(conv_idx[0][0], list):
                    for idx, layer in enumerate(conv_idx):
                        for idx2, sub_layer in enumerate(layer):
                            isinconvidx = isinconvidx or (layer_idx in sub_layer)
                            if layer_idx in sub_layer:
                                mask = conv_mask[idx][idx2][sub_layer.index(layer_idx)]
                else:
                    for idx, layer in enumerate(conv_idx):
                        isinconvidx = isinconvidx or (layer_idx in layer)
                        if layer_idx in layer:
                            mask = conv_mask[idx][layer.index(layer_idx)]

                isinlinearidx = layer_idx in linear_idx
                if not isinconvidx and not isinlinearidx:
                    continue
                # print(type(p), type(p).__name__)
                if p.grad is None:
                    continue
                w_norm = p.data.norm(2, 1).unsqueeze(1)
                # print(w_norm)
                d_p = p.grad.data * w_norm
                # d_p = p.grad.data * w_norm * lr
                # if weight_decay != 0:
                # 	d_p.add_(weight_decay * lr, p.data * (1-mask))
                if momentum != 0:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).cuda()
                        # buf.mul_(momentum).add_(d_p.mul(w_norm))
                        buf = (buf * momentum + d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        # buf.mul_(momentum).add_((1 - dampening), d_p.mul(w_norm))
                        buf = (buf * momentum + (1 - momentum) * d_p)
                    if nesterov:
                        # d_p = d_p.add(momentum, buf)
                        d_p = d_p + momentum * buf * (1 - mask)
                    else:
                        d_p = buf * (1 - mask) + mask * d_p
                    # d_p = buf

                p.grad.data = d_p

        return loss

    def fake_scale_mom_step_real_nomask(self, conv_idx, linear_idx, conv_mask, linear_mask, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            # dampening = momentum
            nesterov = group['nesterov']

            for layer_idx, p in enumerate(group['params']):
                isinconvidx = False

                if isinstance(conv_idx[0][0], list):
                    for idx, layer in enumerate(conv_idx):
                        for idx2, sub_layer in enumerate(layer):
                            isinconvidx = isinconvidx or (layer_idx in sub_layer)
                            if layer_idx in sub_layer:
                                mask = conv_mask[idx][idx2][sub_layer.index(layer_idx)]
                else:
                    for idx, layer in enumerate(conv_idx):
                        isinconvidx = isinconvidx or (layer_idx in layer)
                        if layer_idx in layer:
                            mask = conv_mask[idx][layer.index(layer_idx)]

                isinlinearidx = layer_idx in linear_idx
                if not isinconvidx and not isinlinearidx:
                    continue
                # print(type(p), type(p).__name__)
                if p.grad is None:
                    continue
                w_norm = p.data.norm(2, 1).unsqueeze(1)
                # print(w_norm)
                d_p = p.grad.data * w_norm * lr
                # d_p = p.grad.data * w_norm * lr
                # if weight_decay != 0:
                # 	d_p.add_(weight_decay * lr, p.data * (1-mask))
                if momentum != 0:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).cuda()
                        # buf.mul_(momentum).add_(d_p.mul(w_norm))
                        buf = (buf * momentum + d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        # buf.mul_(momentum).add_((1 - dampening), d_p.mul(w_norm))
                        buf = (buf * momentum + (1 - dampening) * d_p)
                    if nesterov:
                        # d_p = d_p.add(momentum, buf)
                        d_p = d_p + momentum * buf
                    else:
                        d_p = buf
                    # d_p = buf

                p.grad.data = d_p

        return loss

    # Difference with fake_scale_mom_step(): no norm at sk
    def fake_nsk_scale_mom_step(self, conv_idx, linear_idx, conv_mask, linear_mask, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            # dampening = momentum
            nesterov = group['nesterov']

            for layer_idx, p in enumerate(group['params']):
                isinconvidx = False

                if isinstance(conv_idx[0][0], list):
                    for idx, layer in enumerate(conv_idx):
                        for idx2, sub_layer in enumerate(layer):
                            isinconvidx = isinconvidx or (layer_idx in sub_layer)
                            if layer_idx in sub_layer:
                                mask = conv_mask[idx][idx2][sub_layer.index(layer_idx)]
                else:
                    for idx, layer in enumerate(conv_idx):
                        isinconvidx = isinconvidx or (layer_idx in layer)
                        if layer_idx in layer:
                            mask = conv_mask[idx][layer.index(layer_idx)]

                isinlinearidx = layer_idx in linear_idx
                if not isinconvidx and not isinlinearidx:
                    continue
                # print(type(p), type(p).__name__)
                if p.grad is None:
                    continue
                w_norm = p.data.norm(2, 1).unsqueeze(1)
                d_p = p.grad.data * lr * w_norm * (1 - mask) + p.grad.data * lr * mask
                # if weight_decay != 0:
                # 	d_p.add_(weight_decay * lr, p.data * (1-mask))
                if momentum != 0:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).cuda()
                        # buf.mul_(momentum).add_(d_p.mul(w_norm))
                        buf = (buf * momentum + d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        # buf.mul_(momentum).add_((1 - dampening), d_p.mul(w_norm))
                        buf = (buf * momentum + (1 - dampening) * d_p)
                    if nesterov:
                        # d_p = d_p.add(momentum, buf)
                        d_p = d_p + momentum * buf * (1 - mask)
                    else:
                        d_p = buf * (1 - mask) + mask * d_p

                p.grad.data = d_p

        return loss

    def fake_nsk_scale_mom_nomask_step(self, conv_idx, linear_idx, conv_mask, linear_mask, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            # dampening = momentum
            nesterov = group['nesterov']

            for layer_idx, p in enumerate(group['params']):
                isinconvidx = False

                if isinstance(conv_idx[0][0], list):
                    for idx, layer in enumerate(conv_idx):
                        for idx2, sub_layer in enumerate(layer):
                            isinconvidx = isinconvidx or (layer_idx in sub_layer)
                            if layer_idx in sub_layer:
                                mask = conv_mask[idx][idx2][sub_layer.index(layer_idx)]
                else:
                    for idx, layer in enumerate(conv_idx):
                        isinconvidx = isinconvidx or (layer_idx in layer)
                        if layer_idx in layer:
                            mask = conv_mask[idx][layer.index(layer_idx)]

                isinlinearidx = layer_idx in linear_idx
                if not isinconvidx and not isinlinearidx:
                    continue
                # print(type(p), type(p).__name__)
                if p.grad is None:
                    continue
                w_norm = p.data.norm(2, 1).unsqueeze(1)
                d_p = p.grad.data * lr * w_norm * (1 - mask) + p.grad.data * lr * mask
                # if weight_decay != 0:
                # 	d_p.add_(weight_decay * lr, p.data * (1-mask))
                if momentum != 0:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).cuda()
                        # buf.mul_(momentum).add_(d_p.mul(w_norm))
                        buf = (buf * momentum + d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        # buf.mul_(momentum).add_((1 - dampening), d_p.mul(w_norm))
                        buf = (buf * momentum + (1 - dampening) * d_p)
                    if nesterov:
                        # d_p = d_p.add(momentum, buf)
                        d_p = d_p + momentum * buf
                    else:
                        d_p = buf

                p.grad.data = d_p

        return loss

    def partial_bp_step(self, conv_idx, linear_idx, closure=None, debug=False, f=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            # dampening = momentum
            nesterov = group['nesterov']

            for layer_idx, p in enumerate(group['params']):
                isinconvidx = False
                if isinstance(conv_idx[0][0], list):
                    for layer in conv_idx:
                        for sub_layer in layer:
                            isinconvidx = isinconvidx or (layer_idx in sub_layer)
                else:
                    for layer in conv_idx:
                        isinconvidx = isinconvidx or (layer_idx in layer)

                if isinconvidx:
                    continue
                isinlinearidx = layer_idx in linear_idx

                if isinlinearidx:
                    continue
                if p.grad is None:
                    continue
                if debug:
                    f.write('L:{},shape:{},w_norm:{},avg_wnorm:{},g_norm:{},avg_gnorm:{}\n'.format(layer_idx,
                                                                                                   p.data.shape,
                                                                                                   p.data.norm(),
                                                                                                   p.data.norm() / p.data.numel(),
                                                                                                   p.grad.data.norm(),
                                                                                                   p.grad.data.norm() / p.grad.data.numel()))
                d_p = p.grad.data * lr
                if weight_decay != 0:
                    d_p.add_(weight_decay * lr, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).cuda()
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_((1 - dampening), d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-1, d_p)
        return loss

    def cntk_partial_bp_step(self, conv_idx, linear_idx, closure=None, debug=False, f=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            # dampening = momentum
            nesterov = group['nesterov']

            for layer_idx, p in enumerate(group['params']):
                isinconvidx = False
                if isinstance(conv_idx[0][0], list):
                    for layer in conv_idx:
                        for sub_layer in layer:
                            isinconvidx = isinconvidx or (layer_idx in sub_layer)
                else:
                    for layer in conv_idx:
                        isinconvidx = isinconvidx or (layer_idx in layer)

                if isinconvidx:
                    continue
                isinlinearidx = layer_idx in linear_idx

                if isinlinearidx:
                    continue
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).cuda()
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_((1 - momentum), d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-lr, d_p)
        return loss

'''