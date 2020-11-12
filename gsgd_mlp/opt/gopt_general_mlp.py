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
from opt.SGD import SGD

import random
import math


class gSGD():

    def __init__(self, models, lr,args, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        self.models = models
        self.lr_0 = self.lr_t = lr
        self.params = list(models.parameters())
        self.args = args
        # if args.opt == 'ec':
        self.mask, self.s_mask_idx,self.s_mask_idx_shape = self.cal_mask(models)
        self.cal_internal_optim()

    def cal_internal_optim(self):
        self.internal_optim = SGD(self.models.parameters(), lr=self.lr_t )

    def cal_mask(self, model):
        self.s_idx = 0
        num_layer = len(self.params)
        num_hidden = num_layer - 1
        mask = []

        tmp = [0, 0]
        for ii in range(num_layer - 1):
            h = self.params[ii].shape[0]
            if h > tmp[1]:
                tmp = [ii, h]


        self.s_idx = tmp[0]
        self.s_h = tmp[1]

        for layer_idx, layer in enumerate(self.params):
            outcome, income = layer.shape
            if layer_idx == 0:
                loc = 'f'
            elif layer_idx == num_layer - 1:
                loc = 'l'
            else:
                loc = 'm'
            # print(  outcome, income, self.s_idx, loc)
            mask.append(self.generate_eye(outcome, income, loc).to(self.args.device))
            # print(layer.data.device, mask[-1].device)

            layer.data = layer.data * (1 - mask[-1]) + mask[-1]
        s_mask_idx = (mask[self.s_idx]!=0).nonzero().transpose(0,1)
        s_mask_idx_shape = mask[self.s_idx].shape

            # torch.sparse.FloatTensor(a, torch.FloatTensor([1, 1, 1, 1])).to_dense()


        return (mask,s_mask_idx, s_mask_idx_shape)


    def recover_s_layer(self,value,idx , shape):
        assert value.device == idx.device
        # print(idx, value)

        if value.device.type == 'cpu':
            return torch.sparse.FloatTensor(idx, value , shape).to_dense().to(self.args.device)
        else:
            # print(value.device, idx.devic)
            return torch.cuda.sparse.FloatTensor(idx, value , shape).to_dense().to(self.args.device)

    def generate_eye(self, out_shape, in_shape, loc='m'):

        if loc == 'f':
            ratio = out_shape // in_shape + 1

            out_idx = list(range(out_shape))
            in_idx = list(range(in_shape )) *ratio
            idx_tmp = list(zip(out_idx, in_idx))
            idx = torch.LongTensor([x  for x in idx_tmp]).transpose(0,1)
            return(
                self.recover_s_layer(
                    idx = idx,
                    value=torch.ones(out_shape),
                    shape=[out_shape , in_shape ])
            )
        elif loc == 'l':


            ratio = in_shape // out_shape + 1

            out_idx = list(range(out_shape))*ratio
            in_idx = list(range(in_shape ))
            idx_tmp = list(zip(out_idx, in_idx))
            idx = torch.LongTensor([x  for x in idx_tmp]).transpose(0,1)
            return(
                self.recover_s_layer(
                    idx = idx,
                    value=torch.ones(in_shape),
                    shape=[out_shape , in_shape ])
            )
        elif  loc == 'm' :
            if in_shape > out_shape:
                ratio = in_shape // out_shape + 1

                out_idx = list(range(out_shape))*ratio
                in_idx = list(range(in_shape ))

            else:

                ratio = out_shape // in_shape + 1

                out_idx = list(range(out_shape))
                in_idx = list(range(in_shape)) * ratio


            idx_tmp = list(zip(out_idx, in_idx))

            idx = torch.LongTensor([x  for x in idx_tmp]).transpose(0,1)
            # print(len(idx_tmp) , len(in_idx) , len(out_idx), len(idx),max(out_shape, in_shape))
            return(
                self.recover_s_layer(
                    idx = idx,
                    value=torch.ones(max(out_shape, in_shape)),
                    shape=[out_shape , in_shape ])
            )

    def cal_R(self, lr, w_red, dw_red, sigmadwd, v_value):
        return (1 - lr * (dw_red  * w_red  - sigmadwd) / (v_value * v_value))

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""

        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def abs_model(self):
        for p in self.params:
            p = p.abs()

    def remain_input(self, x):
        return x.sum(0)

    def remain_output(self, x):
        return x.sum(1)

    def lr_decay(self, decay, epoch, max_epoch, decay_state):
        if decay == 'poly':
            lr = self.lr_0 * (1 - epoch / max_epoch) ** decay_state['power']
        elif decay == 'linear':
            lr = self.lr_0 * (1 - (epoch + 1) / max_epoch)
        elif decay == 'multistep':
            if (epoch + 1) in decay_state['step']:
                lr = self.lr_t * decay_state['gamma']
            else:
                lr = self.lr_t
        elif decay == 'exp':
            lr = self.lr_0 * math.e ** (-decay_state['power'] * epoch)
        elif decay == 'none':
            lr = self.lr_0
        for param_group in self.internal_optim.param_groups:
            param_group['lr'] = lr
        self.lr_t = lr


    def step(self,closure=None):
        if self.args.opt == "ec":
            self.ec_step(closure)
            # self.bp_partial_step(closure)
        elif self.args.opt == 'bp':
            self.bp_step(closure)


    def bp_partial_step(self, closure):
        self.internal_optim.partial_bp_step_w()

    def bp_step(self,closure):
        self.internal_optim.step()


    def ec_step(self, closure=None):
        """Performs a single optimization step."""

        model = self.models
        num_layer = len(self.params)
        mask = self.mask
        lr = self.lr_t

        w_red = []
        w_blue = []
        dw_red = []
        dw_blue = []
        # v_value = torch.ones_like(self.params[0].sum(1))
        # v_value = torch.ones(self.s_h)
        sigmadwd = torch.zeros(self.s_h).to(self.args.device)

        for layer_idx, layer in enumerate(self.params):
            this_mask = mask[layer_idx]
            # if layer_idx == 0:
            #     loc = 'f'
            # elif layer_idx == num_layer - 1:
            #     loc = 'l'
            # else:
            #     loc = 'm'

            # if loc == 'l':
                # w_red.append((layer.data * this_mask).sum(0))
                # w_blue.append((layer.data * (1 - this_mask)))
            if layer_idx != self.s_idx:
                w_blue =  layer.data * (1 - this_mask)
                # dw_red.append((layer.grad.data * this_mask).sum(0))
                # dw_blue.append((layer.grad.data * (1 - this_mask)))
                dw_blue = layer.grad.data * (1 - this_mask)

                if layer_idx < self.s_idx:
                    # sigmadwd[:w_blue[-1].shape[0]] +=  (w_blue[-1] * dw_blue[-1]).sum(1)
                    sigmadwd[:w_blue.shape[0]] +=  (w_blue * dw_blue).sum(1) ## remain output
                elif layer_idx > self.s_idx:
                    sigmadwd[:w_blue.shape[1]] +=  (w_blue * dw_blue).sum(0) ## remain input
            else:
                v_value = self.remain_output(layer.data * this_mask)
                w_red = v_value
                dw_red = self.remain_output(layer.grad.data * this_mask)





        R = self.cal_R(lr=lr,
                       w_red=w_red,
                       dw_red=dw_red,
                       sigmadwd=sigmadwd,
                       v_value=v_value
                       )
        for layer_idx, layer in enumerate(model.parameters()):
            this_mask = mask[layer_idx]

            if layer_idx == self.s_idx:
                layer_is_s = True
            else:
                layer_is_s = False

            if layer_is_s:
                layer.data = (layer.data - lr * layer.grad.data) * (1 - this_mask) + \
                             layer.data * self.recover_s_layer(value=R,
                                                               idx=self.s_mask_idx,
                                                               shape= self.s_mask_idx_shape)

            elif layer_idx > self.s_idx:
                out_shape, in_shape = layer.data.shape
                layer.data = (layer.data -  lr * layer.grad.data / (v_value[:layer.data.shape[1]].view(1, -1) ** 2) ) / (R[:in_shape].view(1, -1)) * (1 - this_mask)+ \
                    layer.data * this_mask

            elif layer_idx < self.s_idx:
                out_shape, in_shape = layer.data.shape

                layer.data = (layer.data -  lr * layer.grad.data / (v_value[:layer.data.shape[0]].view(-1, 1) ** 2) )/ (R[:out_shape].view(-1, 1)) *(1-this_mask) + \
                    layer.data * this_mask


