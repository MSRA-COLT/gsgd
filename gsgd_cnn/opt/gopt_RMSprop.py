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
from opt.Adam import Adam
from opt.RMSprop import RMSprop
from opt.gopt_RMSprop_method import  gSGD_RMSprop_method

import random
import math


class gSGD_RMSprop(gSGD_RMSprop_method):
    ## need args   ec_layer_type device opt

    def __init__(self, models, lr, args, momentum=0, dampening=0, alpha=0.99,eps=1e-8, bn =False,
                 weight_decay=0, nesterov=False):
        gSGD_RMSprop_method.__init__(self,args=args)

        self.models = models
        # print([(x[0], x[1].shape )for x in list(models.named_parameters())])
        self.lr_0 = self.lr_t = lr
        self.params = list(models.parameters())
        self.args = args
        # self.momentum = momentum
        self.alpha=alpha
        self.eps = eps
        # self.momentum_buffer = {}
        self.square_avg = {'cnn' : [],
                           'mlp' : []}
        self.bn = bn

        if self.bn == False:
            assert self.args.rmsprop_method == 'noweight'

        self.ec_layer = self.cal_ec_layer()
        self.mask = self.cal_mask(models)
        self.cal_internal_optim()
        self.internal_optim.cal_ec_layer(self.ec_layer)
        self.init_square_avg()


    def cal_ec_layer(self):
        if self.args.ec_layer_type == 0:
            ec_layer = self.models.ec_layer
        elif self.args.ec_layer_type == 1:
            ec_layer = self.cal_ec_layer_type1()
        elif self.args.ec_layer_type ==2 :
            ec_layer = self.cal_ec_layer_type2()
        return ec_layer

    def cal_ec_layer_type1(self):
        named_params = self.models.named_parameters()
        ec_layer = [[]]
        for layer_idx, (ii, jj) in enumerate(named_params):

            if 'weight' in ii:
                ec_layer[0].append(layer_idx)
        return ec_layer

    def  cal_ec_layer_type2(self):


        ec_layer = {
            'cnn' : [ [0,2,4] ],
            'mlp' : [  ],
        }

        return ec_layer


    def cal_internal_optim(self):
        if self.args.inner_opt == 'sgd':
            self.internal_optim = SGD(self.models.parameters(), lr=self.lr_t)
        elif self.args.inner_opt =='adam':
            self.internal_optim = Adam(self.models.parameters(), lr=self.lr_t)
        elif self.args.inner_opt =='rmsprop':
            self.internal_optim = RMSprop(self.models.parameters(), lr=self.lr_t)
        else:
            assert 1==2

    def cal_mask(self, model):
        self.num_layer = len(self.params)
        mask = {'cnn':[],
                'mlp':[]}
        s_h = {'cnn':[],
                'mlp':[]}
        s_idx = {'cnn':[],
                'mlp':[]}


        for blocks_type, blocks_idx in self.ec_layer.items():
            if blocks_type =='cnn':

                if len(blocks_idx) ==0:
                    continue
                mask_cnn = []

                s_idx_cnn = []
                s_h_cnn = []
                for block in blocks_idx:
                    layer_mask = []
                    num_layer_in_block = len(block)

                    tmp = [0, 0]
                    for ii in range(num_layer_in_block - 1):
                        h = self.params[block[ii]].shape[0]
                        if h > tmp[1]:
                            tmp = [ii, h]

                    layer_s_idx = tmp[0]  # sub layer index (output size max)
                    layer_s_h = tmp[1]  # max output size

                    for ec_layer_idx, layer_idx in enumerate(block):
                        layer = self.params[layer_idx]
                        outcome, income,shape_2, shape_3 = layer.shape
                        if ec_layer_idx == 0:
                            loc = 'f'
                        elif ec_layer_idx == num_layer_in_block - 1:
                            loc = 'l'
                        else:
                            loc = 'm'

                        mask_tmp = self.generate_eye_cnn(outcome, income,shape_2, shape_3, loc).to(self.args.device)
                        layer.data = layer.data * (1 - mask_tmp) + mask_tmp
                        layer_mask.append(mask_tmp)

                    mask_cnn.append(layer_mask)
                    s_h_cnn.append(layer_s_h)
                    s_idx_cnn.append(layer_s_idx)
                mask['cnn'] = mask_cnn
                s_h['cnn'] = s_h_cnn
                s_idx['cnn'] = s_idx_cnn
            elif blocks_type =='mlp' :

                if len(blocks_idx) ==0:
                    continue
                mask_mlp = []

                s_idx_mlp = []
                s_h_mlp = []
                for block in blocks_idx:
                    layer_mask = []
                    num_layer_in_block = len(block)

                    tmp = [0, 0]
                    for ii in range(num_layer_in_block - 1):
                        h = self.params[block[ii]].shape[0]
                        if h > tmp[1]:
                            tmp = [ii, h]

                    layer_s_idx = tmp[0]  # sub layer index (output size max)
                    layer_s_h = tmp[1]  # max output size

                    for ec_layer_idx, layer_idx in enumerate(block):
                        layer = self.params[layer_idx]
                        outcome, income = layer.shape
                        if ec_layer_idx == 0:
                            loc = 'f'
                        elif ec_layer_idx == num_layer_in_block - 1:
                            loc = 'l'
                        else:
                            loc = 'm'

                        mask_tmp = self.generate_eye_mlp(outcome, income, loc).to(self.args.device)
                        layer.data = layer.data * (1 - mask_tmp) + mask_tmp
                        layer_mask.append(mask_tmp)

                    mask_mlp.append(layer_mask)
                    s_h_mlp.append(layer_s_h)
                    s_idx_mlp.append(layer_s_idx)
                mask['mlp'] = mask_mlp
                s_h['mlp'] = s_h_mlp
                s_idx['mlp'] = s_idx_mlp

        self.s_h = s_h
        self.s_idx = s_idx
        return (mask)




    def step(self, closure=None):
        if self.args.ecopt == "ec":
            # self.pre_step()
            self.ec_step(closure)
            self.bp_partial_step(closure)
        elif self.args.ecopt == 'bp':
            self.bp_step(closure)



    def bp_partial_step(self, closure):
        self.internal_optim.partial_bp_step()

    def bp_step(self, closure):
        self.internal_optim.step()

    def ec_step(self, closure=None):
        """Performs a single optimization step."""
        # lr = self.lr_t

        for blocks_type, blocks_idx in self.ec_layer.items():
            if blocks_type =='cnn':
                if len(blocks_idx) ==0:
                    continue
                self.ec_step_cnn(blocks_idx)
            elif blocks_type =='mlp':
                if len(blocks_idx) ==0:
                    continue
                self.ec_step_mlp(blocks_idx)
            else:
                assert 1==2


    def init_square_avg(self):
        for blocks_type, blocks_idx in self.ec_layer.items():
            if blocks_type =='cnn':
                if len(blocks_idx) ==0:
                    continue
                for block in blocks_idx:
                    square_avg_tmp=[]
                    for ec_layer_idx, layer_idx in enumerate(block):
                        square_avg_tmp.append(torch.zeros_like(self.params[layer_idx]))
                    self.square_avg['cnn'].append(square_avg_tmp)

            elif blocks_type =='mlp' :
                if len(blocks_idx) ==0:
                    continue
                for block in blocks_idx:
                    square_avg_tmp = []
                    for ec_layer_idx, layer_idx in enumerate(block):
                        square_avg_tmp.append(torch.zeros_like(self.params[layer_idx]))
                    self.square_avg['mlp'].append(square_avg_tmp)



