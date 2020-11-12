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

class gSGD():

    def __init__(self, models, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        # if lr is not required and lr < 0.0:
        #     raise ValueError("Invalid learning rate: {}".format(lr))
        # if momentum < 0.0:
        #     raise ValueError("Invalid momentum value: {}".format(momentum))
        # if weight_decay < 0.0:
        #     raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.models = models
        self.lr = lr
        self.params = list(models.parameters())
        self.mask = self.cal_mask(models)

    def cal_mask(self, model):
        num_layer = len(self.params )
        mask = []
        for layer_idx, layer in enumerate(model.parameters()):
            outcome, income = layer.shape
            if layer_idx == 0:
                loc = 'f'
            elif layer_idx == num_layer - 1:
                loc = 'l'
            # print(layer_idx , num_layer)
            mask.append(self.generate_eye(outcome, income, loc))

            layer.data = layer.data * (1 - mask[-1]) + mask[-1]
        return (mask)

    def generate_eye(self, n, m=None, loc='m'):
        if m is None:
            return torch.eye(n)
        elif (n <= m and loc == 'f') or (n >= m and loc == 'l'):
            return torch.eye(n, m)
        elif (n > m and loc == 'f') or (n < m and loc == 'l'):
            return torch.cat([torch.eye(m)] * (n // m + 1), 0)[:n, :m]

    def cal_R(self, lr, w_red, dw_red, sigmadwd, v_value):
        return (1 - lr * (dw_red[0] * w_red[0] - sigmadwd) / (v_value * v_value))

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""

        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def abs_model(self):
        for p in self.params:
            p = p.abs()


    def step(self, closure=None):
        """Performs a single optimization step."""

        model = self.models
        num_layer = len(self.params)
        mask = self.mask
        lr = self.lr

        w_red = []
        w_blue = []
        dw_red = []
        dw_blue = []
        v_value = torch.ones_like(self.params[0].sum(1))
        sigmadwd = 0

        for layer_idx, layer in enumerate(self.params):
            this_mask = mask[layer_idx]
            if layer_idx == 0:
                loc = 'f'
            elif layer_idx == num_layer - 1:
                loc = 'l'
            else:
                loc = 'm'

            if loc == 'f':
                w_red.append((layer.data * this_mask).sum(1))
                w_blue.append((layer.data * (1 - this_mask)))
                dw_red.append((layer.grad.data * this_mask).sum(1))
                dw_blue.append((layer.grad.data * (1 - this_mask)))
            else:
                w_red.append((layer.data * this_mask).sum(0))
                w_blue.append((layer.data * (1 - this_mask)))
                dw_blue.append((layer.grad.data * (1 - this_mask)))
                sigmadwd = sigmadwd + (w_blue[-1] * dw_blue[-1]).sum(0)
            v_value = v_value * w_red[-1]

        R = self.cal_R(lr=lr,
                  w_red=w_red,
                  dw_red=dw_red,
                  sigmadwd=sigmadwd,
                  v_value=v_value
                  )
        for layer_idx, layer in enumerate(model.parameters()):
            this_mask = mask[layer_idx]
            if layer_idx == 0:
                loc = 'f'
            elif layer_idx == num_layer - 1:
                loc = 'l'
            else:
                loc = 'm'

            if loc == 'f':
                tmp_dw_first_blue = (1 - lr * layer.grad.data / layer.data) * (1 - this_mask)
                tmp_dw_first_red = torch.diag(R) * this_mask

                delta_w = tmp_dw_first_blue + tmp_dw_first_red

            else:
                tmp_blue = (1 - lr * layer.grad.data / (v_value ** 2) / layer.data) * (1 - this_mask)
                tmp_red = this_mask
                delta_w = tmp_blue + tmp_red

            layer.data = layer.data * delta_w


