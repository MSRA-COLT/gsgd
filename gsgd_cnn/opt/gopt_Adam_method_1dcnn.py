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



class gSGD_Adam_method():
    ## need args   ec_layer_type device opt

    def __init__(self,   args):
        self.args = args
        self.params = None
        self.lr_0 = None
        self.lr_t = None
        self.internal_optim = None
        self.betas = None
        self.exp_avg = None
        self.exp_avg_sq = None
        self.itr_num = None
        self.eps = None



    # def ec_step(self, closure=None):
    #     pass



    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""

        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def abs_model(self):
        for p in self.params:
            p = p.abs()

    def recover_s_layer(self,value,idx , shape):
        assert value.device == idx.device
        if value.device.type == 'cpu':
            return torch.sparse.FloatTensor(idx, value , shape).to_dense().to(self.args.device)
        else:
            return torch.cuda.sparse.FloatTensor(idx, value , shape).to_dense().to(self.args.device)

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



    def remain_input_cnn(self, x):
        return x.sum(0).sum(1)

    def remain_output_cnn(self, x):
        return x.sum(1).sum(1)

    def remain_input_mlp(self, x):
        return x.sum(0)

    def remain_output_mlp(self, x):
        return x.sum(1)





    def generate_eye_cnn(self, out_shape, in_shape, shape_2,  loc ):

        if loc == 'f':

            shape_2_center = shape_2 //2
            # shape_3_center = shape_3 // 2

            ratio = out_shape // in_shape + 1

            out_idx = list(range(out_shape))
            in_idx = list(range(in_shape )) *ratio
            idx_tmp = list(zip(out_idx, in_idx))
            idx = torch.LongTensor([(*x , shape_2_center )  for x in idx_tmp]).transpose(0,1)
            return(
                self.recover_s_layer(
                    idx = idx,
                    value=torch.ones(out_shape),
                    shape=[out_shape , in_shape , shape_2 ])
            )

        elif loc == 'l':

            shape_2_center = shape_2 //2
            # shape_3_center = shape_3 // 2

            ratio = in_shape // out_shape + 1

            out_idx = list(range(out_shape))*ratio
            in_idx = list(range(in_shape ))
            idx_tmp = list(zip(out_idx, in_idx))
            idx = torch.LongTensor([(*x , shape_2_center )  for x in idx_tmp]).transpose(0,1)
            return(
                self.recover_s_layer(
                    idx = idx,
                    value=torch.ones(in_shape),
                    shape=[out_shape , in_shape , shape_2 ])
            )
        else:

            shape_2_center = shape_2 //2
            # shape_3_center = shape_3 // 2

            if in_shape > out_shape:
                ratio = in_shape // out_shape + 1

                out_idx = list(range(out_shape))*ratio
                in_idx = list(range(in_shape ))
            else:
                shape_2_center = shape_2 // 2
                # shape_3_center = shape_3 // 2

                ratio = out_shape // in_shape + 1

                out_idx = list(range(out_shape))
                in_idx = list(range(in_shape)) * ratio

            idx_tmp = list(zip(out_idx, in_idx))
            idx = torch.LongTensor([(*x , shape_2_center )  for x in idx_tmp]).transpose(0,1)
            return(
                self.recover_s_layer(
                    idx = idx,
                    value=torch.ones(max(out_shape, in_shape)),
                    shape=[out_shape , in_shape , shape_2 ])
            )






    def generate_eye_mlp(self, out_shape, in_shape, loc ):

        if loc == 'f':
            ratio = out_shape // in_shape + 1

            out_idx = list(range(out_shape))
            in_idx = list(range(in_shape)) * ratio
            idx_tmp = list(zip(out_idx, in_idx))
            idx = torch.LongTensor([x for x in idx_tmp]).transpose(0, 1)
            return (
                self.recover_s_layer(
                    idx=idx,
                    value=torch.ones(out_shape),
                    shape=[out_shape, in_shape])
            )
        elif loc == 'l':

            ratio = in_shape // out_shape + 1

            out_idx = list(range(out_shape)) * ratio
            in_idx = list(range(in_shape))
            idx_tmp = list(zip(out_idx, in_idx))
            idx = torch.LongTensor([x for x in idx_tmp]).transpose(0, 1)
            return (
                self.recover_s_layer(
                    idx=idx,
                    value=torch.ones(in_shape),
                    shape=[out_shape, in_shape])
            )
        elif loc == 'm':
            if in_shape > out_shape:
                ratio = in_shape // out_shape + 1

                out_idx = list(range(out_shape)) * ratio
                in_idx = list(range(in_shape))

            else:

                ratio = out_shape // in_shape + 1

                out_idx = list(range(out_shape))
                in_idx = list(range(in_shape)) * ratio

            idx_tmp = list(zip(out_idx, in_idx))

            idx = torch.LongTensor([x for x in idx_tmp]).transpose(0, 1)
            return (
                self.recover_s_layer(
                    idx=idx,
                    value=torch.ones(max(out_shape, in_shape)),
                    shape=[out_shape, in_shape])
            )


    def cal_R_cnn(self, lr, w_red, dw_red, sigmadwd, v_value, w_norm, mask,exp_avg, exp_avg_sq):

        bias_correction1 = 1 - self.beta1 ** self.itr_num
        bias_correction2 = 1 - self.beta2 ** self.itr_num



        if self.args.adam_method =='weight':
            lr1 = lr * np.sqrt(bias_correction2) / bias_correction1

            lr2 = (lr1 / ( torch.sqrt(exp_avg_sq) +  self.eps ) * mask).sum(1).sum(1)

            res = 1 - lr2 *  ( exp_avg* mask).sum(1).sum(1) /  v_value

        elif self.args.adam_method =='noweight':

            lr1 = lr * np.sqrt(bias_correction2) / bias_correction1

            lr2 = (lr1 / ( torch.sqrt(exp_avg_sq)*w_norm +  self.eps ) * mask).sum(1).sum(1)

            res = 1 - lr2 * ( exp_avg * w_norm * mask).sum(1).sum(1)  / v_value

        return(res)


    def cal_R_mlp(self, lr, w_red, dw_red, sigmadwd, v_value, mask,exp_avg, exp_avg_sq):

        bias_correction1 = 1 - self.beta1 ** self.itr_num
        bias_correction2 = 1 - self.beta2 ** self.itr_num


        lr1 = lr * np.sqrt(bias_correction2) / bias_correction1
        lr2 = (lr1 / ( torch.sqrt(exp_avg_sq) +  self.eps ) * mask).sum(1)
        res = 1 - lr2 *  ( exp_avg* mask).sum(1) /  v_value
        return(res)




    def ec_step_cnn(self, blocks_idx, closure=None):
        """Performs a single optimization step."""
        beta1, beta2 = self.betas

        lr_t = self.lr_t

        mask = self.mask['cnn']
        s_idx = self.s_idx['cnn']
        s_h = self.s_h['cnn']


        for block_idx, block in enumerate(blocks_idx):

            this_block_mask = mask[block_idx]
            this_block_s_h = s_h[block_idx]
            this_block_s_idx = s_idx[block_idx]

            sigmadwd = torch.zeros(this_block_s_h).to(self.args.device)
            w_norm =[]

            for ec_layer_idx, layer_idx in enumerate(block):
                layer = self.params[layer_idx]

                this_mask_sub_layer = this_block_mask[ec_layer_idx]
                if self.bn == True:
                    w_norm.append(layer.data.norm(2, 1).unsqueeze(1))
                else:
                    w_norm.append(1)
                if   ec_layer_idx != this_block_s_idx:
                    w_blue = layer.data * (1 - this_mask_sub_layer)
                    dw_blue = layer.grad.data * (1 - this_mask_sub_layer)
                    if ec_layer_idx < this_block_s_idx:
                        sigmadwd[:w_blue.shape[0]] += self.remain_output_cnn(w_blue * dw_blue)
                    elif ec_layer_idx > this_block_s_idx:
                        sigmadwd[:w_blue.shape[1]] +=  self.remain_input_cnn(w_blue * dw_blue)



                elif ec_layer_idx == this_block_s_idx:
                    v_value = self.remain_output_cnn(layer.data * this_mask_sub_layer)
                    w_red = v_value
                    dw_red = self.remain_output_cnn(layer.grad.data * this_mask_sub_layer)

            ## start calculate exp_avg and exp_avg_sq  buffer
            for ec_layer_idx, layer_idx in enumerate(block):
                layer = self.params[layer_idx]
                out_shape, in_shape, shape_2 = layer.data.shape

                this_mask_sub_layer = this_block_mask[ec_layer_idx]

                # not special layer
                if ec_layer_idx != this_block_s_idx:
                    dw_blue = layer.grad.data * (1 - this_mask_sub_layer)

                    ## lower layer
                    if ec_layer_idx < this_block_s_idx:

                        if self.args.adam_method =='weight' :
                            self.exp_avg['cnn'][block_idx][ec_layer_idx] = \
                                (beta1 *
                                    self.exp_avg['cnn'][block_idx][ec_layer_idx]
                                + (1 - beta1) *
                                    (dw_blue * w_norm[ec_layer_idx]
                                      / v_value[:out_shape].unsqueeze(1).unsqueeze(1)
                                     + self.weight_decay * layer.data * v_value[:out_shape].unsqueeze(1).unsqueeze(1)
                                     )  ) \
                                 * (1 - this_mask_sub_layer)      + this_mask_sub_layer

                            self.exp_avg_sq['cnn'][block_idx][ec_layer_idx] = \
                                (beta2 *
                                    self.exp_avg_sq['cnn'][block_idx][ec_layer_idx]
                                + (1 - beta2) *
                                    (dw_blue * w_norm[ec_layer_idx]
                                      / v_value[:out_shape].unsqueeze(1).unsqueeze(1)
                                     + self.weight_decay * layer.data * v_value[:out_shape].unsqueeze(1).unsqueeze(1)) ** 2) \
                                 * (1 - this_mask_sub_layer)      + this_mask_sub_layer



                        elif self.args.adam_method =='noweight':


                            self.exp_avg['cnn'][block_idx][ec_layer_idx] = \
                                (beta1 *
                                        self.exp_avg['cnn'][block_idx][ec_layer_idx]
                                + (1 - beta1) *
                                        (dw_blue / v_value[:out_shape].unsqueeze(1).unsqueeze(1)
                                         + self.weight_decay * layer.data * v_value[:out_shape].unsqueeze(1).unsqueeze(1)))\
                                *(1 - this_mask_sub_layer)      + this_mask_sub_layer

                            self.exp_avg_sq['cnn'][block_idx][ec_layer_idx] = \
                                (beta2 *
                                        self.exp_avg_sq['cnn'][block_idx][ec_layer_idx]
                                + (1 - beta2) *
                                        (dw_blue / v_value[:out_shape].unsqueeze(1).unsqueeze(1)
                                         + self.weight_decay * layer.data * v_value[:out_shape].unsqueeze(1).unsqueeze(1))**2)\
                                *(1 - this_mask_sub_layer)      + this_mask_sub_layer


                    # upper layer
                    elif ec_layer_idx > this_block_s_idx:
                        if self.args.adam_method == 'weight':

                            self.exp_avg['cnn'][block_idx][ec_layer_idx] = \
                                (beta1 *
                                     self.exp_avg['cnn'][block_idx][ec_layer_idx]
                                + (1 - beta1) *
                                     (dw_blue* w_norm[ec_layer_idx]
                                      / v_value[:in_shape].unsqueeze(1).unsqueeze(0)
                                      + self.weight_decay * layer.data * v_value[:in_shape].unsqueeze(1).unsqueeze(0)))\
                                *(1 - this_mask_sub_layer)     + this_mask_sub_layer

                            self.exp_avg_sq['cnn'][block_idx][ec_layer_idx] = \
                                (beta2 *
                                     self.exp_avg_sq['cnn'][block_idx][ec_layer_idx]
                                + (1 - beta2) *
                                     (dw_blue* w_norm[ec_layer_idx]
                                      / v_value[:in_shape].unsqueeze(1).unsqueeze(0)
                                      + self.weight_decay * layer.data * v_value[:in_shape].unsqueeze(1).unsqueeze(0))**2 )\
                                *(1 - this_mask_sub_layer)     + this_mask_sub_layer

                        elif self.args.adam_method == 'noweight':
                            self.exp_avg['cnn'][block_idx][ec_layer_idx] = \
                                (beta1 *
                                    self.exp_avg['cnn'][block_idx][ec_layer_idx]
                                + (1 - beta1) *
                                    (dw_blue / v_value[:in_shape].unsqueeze(1).unsqueeze(0)
                                     + self.weight_decay * layer.data * v_value[:in_shape].unsqueeze(1).unsqueeze(0))) \
                                *(1 - this_mask_sub_layer)    + this_mask_sub_layer

                            self.exp_avg_sq['cnn'][block_idx][ec_layer_idx] = \
                                (beta2 *
                                    self.exp_avg_sq['cnn'][block_idx][ec_layer_idx]
                                + (1 - beta2) *
                                    (dw_blue / v_value[:in_shape].unsqueeze(1).unsqueeze(0)
                                     + self.weight_decay * layer.data * v_value[:in_shape].unsqueeze(1).unsqueeze(0))**2) \
                                *(1 - this_mask_sub_layer)    + this_mask_sub_layer

                # special layer
                elif ec_layer_idx == this_block_s_idx:
                    # dw_red = self.remain_output_cnn(layer.grad.data * this_mask_sub_layer)
                    dw_blue = layer.grad.data * (1 - this_mask_sub_layer)
                    if self.args.adam_method == 'weight':

                        tmp_exp_avg_blue =    \
                        (beta1 *
                            self.exp_avg['cnn'][block_idx][ec_layer_idx]
                        + (1 - beta2) *
                            (dw_blue * w_norm[ec_layer_idx] + self.weight_decay*layer.data ) ) \
                        * (1 - this_mask_sub_layer)

                        tmp_exp_avg_sq_blue =    \
                        (beta2 *
                            self.exp_avg_sq['cnn'][block_idx][ec_layer_idx]
                        + (1 - beta2) *
                            (dw_blue * w_norm[ec_layer_idx] + self.weight_decay*layer.data )**2) \
                        * (1 - this_mask_sub_layer)

                        tmp_exp_avg_red = \
                            (beta1 *
                                self.exp_avg['cnn'][block_idx][ec_layer_idx]
                            + (1 - beta1) *
                                (
                                     (     ( w_red*dw_red   - sigmadwd)  /  ((v_value)
                                            * (w_norm[ec_layer_idx]*this_mask_sub_layer).sum(1).sum(1) )
                                      )
                                ).unsqueeze(1).unsqueeze(1)) \
                            * this_mask_sub_layer

                        tmp_exp_avg_sq_red = \
                            (beta2 *
                                self.exp_avg_sq['cnn'][block_idx][ec_layer_idx]
                            + (1 - beta2) *
                                (
                                     (     ( w_red*dw_red   - sigmadwd)  /  ((v_value)
                                            * (w_norm[ec_layer_idx]*this_mask_sub_layer).sum(1).sum(1))
                                      )**2
                                ).unsqueeze(1).unsqueeze(1)) \
                            * this_mask_sub_layer
                        self.exp_avg['cnn'][block_idx][ec_layer_idx] = tmp_exp_avg_blue + tmp_exp_avg_red
                        self.exp_avg_sq['cnn'][block_idx][ec_layer_idx] = tmp_exp_avg_sq_blue + tmp_exp_avg_sq_red
                    elif self.args.adam_method == 'noweight':

                        tmp_exp_avg_blue =   \
                            (beta1 *
                                self.exp_avg['cnn'][block_idx][ec_layer_idx]
                            + (1 - beta1) *
                                (dw_blue+ self.weight_decay*layer.data) ) \
                            * (1 - this_mask_sub_layer)
                        tmp_exp_avg_sq_blue =   \
                            (beta2 *
                                self.exp_avg_sq['cnn'][block_idx][ec_layer_idx]
                            + (1 - beta2) *
                                (dw_blue+ self.weight_decay*layer.data )**2) \
                            * (1 - this_mask_sub_layer)
                        tmp_exp_avg_red = \
                            (beta1 *
                                self.exp_avg['cnn'][block_idx][ec_layer_idx]
                            + (1 - beta1) *
                                ((( w_red*dw_red   - sigmadwd) /  (v_value) )  ).unsqueeze(1).unsqueeze(1)) \
                            * this_mask_sub_layer
                        tmp_exp_avg_sq_red = \
                            (beta2 *
                                self.exp_avg_sq['cnn'][block_idx][ec_layer_idx]
                            + (1 - beta2) *
                                ((( w_red*dw_red   - sigmadwd) /  (v_value) )**2  ).unsqueeze(1).unsqueeze(1)) \
                            * this_mask_sub_layer
                        self.exp_avg['cnn'][block_idx][ec_layer_idx] = tmp_exp_avg_blue + tmp_exp_avg_red
                        self.exp_avg_sq['cnn'][block_idx][ec_layer_idx] = tmp_exp_avg_sq_blue + tmp_exp_avg_sq_red

            R = self.cal_R_cnn(
                                lr=lr_t,
                                w_red=w_red,
                                dw_red=dw_red,
                                sigmadwd=sigmadwd,
                                v_value=v_value,
                                w_norm=w_norm[this_block_s_idx],
                                mask=this_block_mask[this_block_s_idx],
                                exp_avg= self.exp_avg['cnn'][block_idx][this_block_s_idx],
                                exp_avg_sq= self.exp_avg_sq['cnn'][block_idx][this_block_s_idx]
                                )

            for ec_layer_idx, layer_idx in enumerate(block):
                this_mask = this_block_mask[ec_layer_idx]
                layer = self.params[layer_idx]
                out_shape, in_shape, shape_2  = layer.data.shape

                this_mask_sub_layer_red = (this_mask == 1).to(torch.float32)
                this_mask_sub_layer_blue = (this_mask == 0).to(torch.float32)

                bias_correction1 = 1 - self.beta1 ** self.itr_num
                bias_correction2 = 1 - self.beta2 ** self.itr_num
                lr1 = lr_t * np.sqrt(bias_correction2) / bias_correction1

                if self.args.adam_method == 'weight':
                    lr2 =  lr1 / ( torch.sqrt(self.exp_avg_sq['cnn'][block_idx][ec_layer_idx]) + self.eps )
                elif self.args.adam_method == 'noweight':
                    # print(type(w_norm))
                    lr2 = lr1 / (torch.sqrt(self.exp_avg_sq['cnn'][block_idx][ec_layer_idx])*w_norm[ec_layer_idx] + self.eps)

                lr = lr2
                weightornot = self.args.adam_method == 'weight'
                this_exp_avg_sub_layer = self.exp_avg['cnn'][block_idx][ec_layer_idx]
                # if weightornot
                if   self.args.adam_method == 'weight':
                    this_wnorm_sub_layer = 1
                else:
                    this_wnorm_sub_layer = w_norm[ec_layer_idx]

                if ec_layer_idx == this_block_s_idx :
                    layer_is_s = True
                else:
                    layer_is_s = False

                if layer_is_s:
                    this_R_value = (R).unsqueeze(1).unsqueeze(1)

                    tmp_red = (
                                      layer.data * this_R_value
                              ) * this_mask_sub_layer_red

                    tmp_blue_first = (
                                             layer.data - lr * this_exp_avg_sub_layer * this_wnorm_sub_layer
                                     ) * this_mask_sub_layer_blue

                    layer.data = tmp_blue_first + tmp_red

                elif ec_layer_idx > this_block_s_idx:
                    tmp_blue =  (layer.data -
                                 lr * this_exp_avg_sub_layer * this_wnorm_sub_layer
                                 /  (v_value[:in_shape]  ).unsqueeze(1).unsqueeze(0)
                                 ) \
                                /(R[:in_shape]).unsqueeze(1).unsqueeze(0) * this_mask_sub_layer_blue

                    tmp_red = layer.data * this_mask_sub_layer_red

                    layer.data = tmp_blue + tmp_red

                elif ec_layer_idx < this_block_s_idx:
                    tmp_blue =  (layer.data -
                                 lr * this_exp_avg_sub_layer  * this_wnorm_sub_layer
                                 /  (v_value[:out_shape]  ).unsqueeze(1).unsqueeze(1)
                                 )  \
                                 /(R[:out_shape]).unsqueeze(1).unsqueeze(1) * this_mask_sub_layer_blue

                    tmp_red = layer.data * this_mask_sub_layer_red

                    layer.data = tmp_blue + tmp_red



    def ec_step_mlp(self, blocks_idx, closure=None):
        """Performs a single optimization step."""
        lr_t = self.lr_t
        beta1, beta2 = self.betas

        mask = self.mask['mlp']
        s_idx = self.s_idx['mlp']
        s_h = self.s_h['mlp']

        for block_idx, block in enumerate(blocks_idx):
            this_block_mask = mask[block_idx]
            this_block_s_h = s_h[block_idx]
            this_block_s_idx = s_idx[block_idx]
            # print(this_block_s_h)
            sigmadwd = torch.zeros(this_block_s_h).to(self.args.device)
            num_layer_in_block=len(block)

            for ec_layer_idx, layer_idx in enumerate(block):

                this_mask = this_block_mask[ec_layer_idx]
                layer = self.params[layer_idx]
                if ec_layer_idx == 0:
                    loc = 'f'
                elif ec_layer_idx == num_layer_in_block - 1:
                    loc = 'l'
                else:
                    loc = 'm'
                # print(ec_layer_idx, this_block_s_idx, '!!!!!!!!!!')
                if ec_layer_idx != this_block_s_idx:

                    w_blue =  layer.data * (1 - this_mask)
                    dw_blue = layer.grad.data * (1 - this_mask)


                    if ec_layer_idx < this_block_s_idx:
                        sigmadwd[:w_blue.shape[0]] +=  (w_blue * dw_blue).sum(1) ## remain output
                    elif ec_layer_idx > this_block_s_idx:
                        sigmadwd[:w_blue.shape[1]] +=  (w_blue * dw_blue).sum(0) ## remain input
                else:
                    v_value = self.remain_output_mlp(layer.data * this_mask)
                    w_red = v_value
                    dw_red = self.remain_output_mlp(layer.grad.data * this_mask)

            ## start calculate exp_avg and exp_avg_sq  buffer
            for ec_layer_idx, layer_idx in enumerate(block):
                layer = self.params[layer_idx]
                out_shape, in_shape = layer.data.shape

                this_mask_sub_layer = this_block_mask[ec_layer_idx]

                # not special layer
                if ec_layer_idx != this_block_s_idx:
                    dw_blue = layer.grad.data * (1 - this_mask_sub_layer)

                    ## lower layer
                    if ec_layer_idx < this_block_s_idx:
                        self.exp_avg['mlp'][block_idx][ec_layer_idx] = \
                            (beta1 *
                                    self.exp_avg['mlp'][block_idx][ec_layer_idx]
                            + (1 - beta1) *
                                    (dw_blue / v_value[:out_shape].unsqueeze(1)
                                        + self.weight_decay * layer.data*v_value[:out_shape].unsqueeze(1)
                                     )
                             )*(1 - this_mask_sub_layer)      + this_mask_sub_layer

                        self.exp_avg_sq['mlp'][block_idx][ec_layer_idx] = \
                            (beta2 *
                                    self.exp_avg_sq['mlp'][block_idx][ec_layer_idx]
                            + (1 - beta2) *
                                    (dw_blue / v_value[:out_shape].unsqueeze(1)
                                        + self.weight_decay * layer.data * v_value[:out_shape].unsqueeze(1)
                                     )**2
                             )*(1 - this_mask_sub_layer)      + this_mask_sub_layer


                    # upper layer
                    elif ec_layer_idx > this_block_s_idx:

                        self.exp_avg['mlp'][block_idx][ec_layer_idx] = \
                            (beta1 *
                                self.exp_avg['mlp'][block_idx][ec_layer_idx]
                            + (1 - beta1) *
                                (dw_blue / v_value[:in_shape].unsqueeze(0)
                                 + self.weight_decay * layer.data * v_value[:in_shape].unsqueeze(0)
                                 )) \
                            *(1 - this_mask_sub_layer)    + this_mask_sub_layer

                        self.exp_avg_sq['mlp'][block_idx][ec_layer_idx] = \
                            (beta2 *
                                self.exp_avg_sq['mlp'][block_idx][ec_layer_idx]
                            + (1 - beta2) *
                                (dw_blue / v_value[:in_shape].unsqueeze(0)
                                 + self.weight_decay * layer.data * v_value[:in_shape].unsqueeze(0)
                                 )**2) \
                            *(1 - this_mask_sub_layer)    + this_mask_sub_layer

                # special layer
                elif ec_layer_idx == this_block_s_idx:
                    # dw_red = self.remain_output_cnn(layer.grad.data * this_mask_sub_layer)
                    dw_blue = layer.grad.data * (1 - this_mask_sub_layer)


                    tmp_exp_avg_blue =   \
                        (beta1 *
                            self.exp_avg['mlp'][block_idx][ec_layer_idx]
                        + (1 - beta1) *
                            (dw_blue + self.weight_decay*layer.data) ) \
                        * (1 - this_mask_sub_layer)
                    tmp_exp_avg_sq_blue =   \
                        (beta2 *
                            self.exp_avg_sq['mlp'][block_idx][ec_layer_idx]
                        + (1 - beta2) *
                            (dw_blue + self.weight_decay*layer.data)**2) \
                        * (1 - this_mask_sub_layer)
                    tmp_exp_avg_red = \
                        (beta1 *
                            self.exp_avg['mlp'][block_idx][ec_layer_idx]
                        + (1 - beta1) *
                            ((( w_red*dw_red   - sigmadwd) /  (v_value) )  ).unsqueeze(1)) \
                        * this_mask_sub_layer
                    tmp_exp_avg_sq_red = \
                        (beta2 *
                            self.exp_avg_sq['mlp'][block_idx][ec_layer_idx]
                        + (1 - beta2) *
                            ((( w_red*dw_red   - sigmadwd) /  (v_value) )**2  ).unsqueeze(1)) \
                        * this_mask_sub_layer
                    self.exp_avg['mlp'][block_idx][ec_layer_idx] = tmp_exp_avg_blue + tmp_exp_avg_red
                    self.exp_avg_sq['mlp'][block_idx][ec_layer_idx] = tmp_exp_avg_sq_blue + tmp_exp_avg_sq_red

            R = self.cal_R_mlp(lr=lr_t,
                            w_red=w_red,
                            dw_red=dw_red,
                            sigmadwd=sigmadwd,
                            v_value=v_value,
                            mask=this_block_mask[this_block_s_idx],
                            exp_avg=self.exp_avg['mlp'][block_idx][this_block_s_idx],
                            exp_avg_sq=self.exp_avg_sq['mlp'][block_idx][this_block_s_idx]
                           )
            for ec_layer_idx, layer_idx in enumerate(block):
                this_mask = this_block_mask[ec_layer_idx]
                layer = self.params[layer_idx]

                bias_correction1 = 1 - self.beta1 ** self.itr_num
                bias_correction2 = 1 - self.beta2 ** self.itr_num

                lr1 = lr_t * np.sqrt(bias_correction2) / bias_correction1

                lr2 =  lr1 / (torch.sqrt(self.exp_avg_sq['mlp'][block_idx][ec_layer_idx]) + self.eps)
                lr = lr2
                this_exp_avg_sub_layer = self.exp_avg['mlp'][block_idx][ec_layer_idx]

                if ec_layer_idx == 0:
                    loc = 'f'
                elif ec_layer_idx == num_layer_in_block - 1:
                    loc = 'l'
                else:
                    loc = 'm'

                if ec_layer_idx == this_block_s_idx:
                    layer_is_s = True
                else:
                    layer_is_s = False

                this_mask_sub_layer_red = (this_mask == 1).to(torch.float32)
                this_mask_sub_layer_blue = (this_mask == 0).to(torch.float32)

                if layer_is_s:
                    this_R_value = (R).unsqueeze(1)
                    tmp_red = (
                                      layer.data * this_R_value
                              ) * this_mask_sub_layer_red

                    tmp_blue_first = (
                                             layer.data - lr * this_exp_avg_sub_layer
                                     ) * this_mask_sub_layer_blue

                    layer.data = tmp_blue_first + tmp_red
                    # layer.data = (layer.data - lr * layer.grad.data) * (1 - this_mask) + \
                                 # layer.data * this_R_value * this_mask

                elif ec_layer_idx > this_block_s_idx:
                    out_shape, in_shape = layer.data.shape

                    tmp_blue =  (layer.data -
                                 lr * this_exp_avg_sub_layer
                                 /  (v_value[:in_shape] ).unsqueeze(0)
                                 ) \
                                /(R[:in_shape]).unsqueeze(0) * this_mask_sub_layer_blue

                    tmp_red = layer.data * this_mask_sub_layer_red

                    layer.data = tmp_blue + tmp_red


                elif ec_layer_idx < this_block_s_idx:
                    out_shape, in_shape = layer.data.shape

                    tmp_blue =  (layer.data -
                                 lr * this_exp_avg_sub_layer
                                 /  (v_value[:out_shape]  ).unsqueeze(1)
                                 )  \
                                 /(R[:out_shape]).unsqueeze(1)  * this_mask_sub_layer_blue

                    tmp_red = layer.data * this_mask_sub_layer_red

                    layer.data = tmp_blue + tmp_red
