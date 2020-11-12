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



class gSGD_RMSprop_method():
    ## need args   ec_layer_type device opt

    def __init__(self,   args):
        self.args = args
        self.params = None
        self.lr_0 = None
        self.lr_t = None
        self.internal_optim = None
        self.alpha = None
        self.square_avg = None



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

    # def step(self,closure=None):
    #     if self.args.opt == "ec":
    #         self.ec_step(closure)
    #         self.bp_partial_step(closure)
    #     elif self.args.opt == 'bp':
    #         self.bp_step(closure)

    # def bp_partial_step(self, closure):
    #     self.internal_optim.partial_bp_step_w()
    #
    # def bp_step(self,closure):
    #     self.internal_optim.step()

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
        return x.sum(0).sum(1).sum(1)

    def remain_output_cnn(self, x):
        return x.sum(1).sum(1).sum(1)

    def remain_input_mlp(self, x):
        return x.sum(0)

    def remain_output_mlp(self, x):
        return x.sum(1)





    def generate_eye_cnn(self, out_shape, in_shape, shape_2, shape_3, loc ):

        if loc == 'f':

            shape_2_center = shape_2 //2
            shape_3_center = shape_3 // 2

            ratio = out_shape // in_shape + 1

            out_idx = list(range(out_shape))
            in_idx = list(range(in_shape )) *ratio
            idx_tmp = list(zip(out_idx, in_idx))
            idx = torch.LongTensor([(*x , shape_2_center , shape_3_center)  for x in idx_tmp]).transpose(0,1)
            return(
                self.recover_s_layer(
                    idx = idx,
                    value=torch.ones(out_shape),
                    shape=[out_shape , in_shape , shape_2 , shape_3])
            )

        elif loc == 'l':

            shape_2_center = shape_2 //2
            shape_3_center = shape_3 // 2

            ratio = in_shape // out_shape + 1

            out_idx = list(range(out_shape))*ratio
            in_idx = list(range(in_shape ))
            idx_tmp = list(zip(out_idx, in_idx))
            idx = torch.LongTensor([(*x , shape_2_center , shape_3_center)  for x in idx_tmp]).transpose(0,1)
            return(
                self.recover_s_layer(
                    idx = idx,
                    value=torch.ones(in_shape),
                    shape=[out_shape , in_shape , shape_2 , shape_3])
            )
        else:

            shape_2_center = shape_2 //2
            shape_3_center = shape_3 // 2

            if in_shape > out_shape:
                ratio = in_shape // out_shape + 1

                out_idx = list(range(out_shape))*ratio
                in_idx = list(range(in_shape ))
            else:
                shape_2_center = shape_2 // 2
                shape_3_center = shape_3 // 2

                ratio = out_shape // in_shape + 1

                out_idx = list(range(out_shape))
                in_idx = list(range(in_shape)) * ratio

            idx_tmp = list(zip(out_idx, in_idx))
            idx = torch.LongTensor([(*x , shape_2_center , shape_3_center)  for x in idx_tmp]).transpose(0,1)
            return(
                self.recover_s_layer(
                    idx = idx,
                    value=torch.ones(max(out_shape, in_shape)),
                    shape=[out_shape , in_shape , shape_2 , shape_3])
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


    def cal_R_mlp(self, lr, w_red, dw_red, sigmadwd, v_value, mask,square_avg):

        lr= (lr / (torch.sqrt(square_avg)+self.eps) * mask).sum(1)
        # print(lr.shape, dw_red.shape, v_value.shape, mask.shape, torch.sqrt(square_avg+self.eps).shape)

        return (1 - lr * (dw_red * w_red - sigmadwd) / (v_value * v_value))

    def cal_R_cnn(self, lr, w_red, dw_red, sigmadwd, v_value, w_norm, mask,square_avg):


        if self.args.rmsprop_method == "weight":

            lr = (lr / (torch.sqrt(square_avg ) + self.eps) * mask).sum(1).sum(1).sum(1)
        elif self.args.rmsprop_method ==  "noweight":
            lr = (lr / (torch.sqrt(square_avg)*w_norm + self.eps) * mask).sum(1).sum(1).sum(1)



        res = 1 - lr * (dw_red * w_red - sigmadwd) /  (v_value * v_value) *( w_norm*mask).sum(1).sum(1).sum(1)
        return(res)

    def ec_step_cnn(self, blocks_idx, closure=None):
        """Performs a single optimization step."""

        lr_t = self.lr_t
        #
        # if self.args.inner_opt != "sgd":
        #     lr = 1

        mask = self.mask['cnn']
        s_idx = self.s_idx['cnn']
        s_h = self.s_h['cnn']
        # square_avg = self.square_avg['cnn']


        for block_idx, block in enumerate(blocks_idx):

            this_block_mask = mask[block_idx]
            this_block_s_h = s_h[block_idx]
            this_block_s_idx = s_idx[block_idx]
            # this_block_square_avg = square_avg[blocks_idx]

            sigmadwd = torch.zeros(this_block_s_h).to(self.args.device)
            w_norm =[]

            for ec_layer_idx, layer_idx in enumerate(block):
                layer = self.params[layer_idx]

                # this_square_avg = square_avg[ec_layer_idx]
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

            ## start calculate square_avg  buffer
            for ec_layer_idx, layer_idx in enumerate(block):
                layer = self.params[layer_idx]
                out_shape, in_shape, shape_2, shape_3 = layer.data.shape

                this_mask_sub_layer = this_block_mask[ec_layer_idx]

                if ec_layer_idx != this_block_s_idx:
                    dw_blue = layer.grad.data * (1 - this_mask_sub_layer)

                    if ec_layer_idx < this_block_s_idx:

                        if self.args.rmsprop_method =='weight' :
                            self.square_avg['cnn'][block_idx][ec_layer_idx] = \
                                (self.alpha *
                                    self.square_avg['cnn'][block_idx][ec_layer_idx]
                                + (1 - self.alpha) *
                                    (dw_blue * w_norm[ec_layer_idx]
                                      / v_value[:out_shape].unsqueeze(1).unsqueeze(1).unsqueeze(1)) ** 2) \
                                 * (1 - this_mask_sub_layer)      + this_mask_sub_layer

                        elif self.args.rmsprop_method =='noweight':


                            self.square_avg['cnn'][block_idx][ec_layer_idx] = \
                                (self.alpha *
                                        self.square_avg['cnn'][block_idx][ec_layer_idx]
                                + (1 - self.alpha) *
                                        (dw_blue / v_value[:out_shape].unsqueeze(1).unsqueeze(1).unsqueeze(1))**2
                                 )*(1 - this_mask_sub_layer)      + this_mask_sub_layer

                    elif ec_layer_idx > this_block_s_idx:
                        if self.args.rmsprop_method == 'weight':
                            self.square_avg['cnn'][block_idx][ec_layer_idx] = \
                                (self.alpha *
                                     self.square_avg['cnn'][block_idx][ec_layer_idx]
                                + (1 - self.alpha) *
                                     (dw_blue* w_norm[ec_layer_idx]
                                      / v_value[:in_shape].unsqueeze(1).unsqueeze(1).unsqueeze(0))**2 )\
                                *(1 - this_mask_sub_layer)     + this_mask_sub_layer
                        elif self.args.rmsprop_method == 'noweight':
                            self.square_avg['cnn'][block_idx][ec_layer_idx] = \
                                (self.alpha *
                                    self.square_avg['cnn'][block_idx][ec_layer_idx]
                                + (1 - self.alpha) *
                                    (dw_blue / v_value[:in_shape].unsqueeze(1).unsqueeze(1).unsqueeze(0))**2) \
                                *(1 - this_mask_sub_layer)    + this_mask_sub_layer

                elif ec_layer_idx == this_block_s_idx:
                    # print(block_idx)
                    # dw_red = self.remain_output_cnn(layer.grad.data * this_mask_sub_layer)
                    dw_blue = layer.grad.data * (1 - this_mask_sub_layer)
                    if self.args.rmsprop_method == 'weight':
                        tmp_square_avg_blue =    \
                        (self.alpha *
                            self.square_avg['cnn'][block_idx][ec_layer_idx]
                        + (1 - self.alpha) *
                            (dw_blue * w_norm[ec_layer_idx])**2) \
                        * (1 - this_mask_sub_layer)
                        tmp_square_avg_red = \
                            (self.alpha *
                                self.square_avg['cnn'][block_idx][ec_layer_idx]
                            + (1 - self.alpha) *
                                (
                                     (     ( w_red*dw_red   - sigmadwd)  /  ((v_value)
                                            * (w_norm[ec_layer_idx]*this_mask_sub_layer).sum(1).sum(1).sum(1) )
                                      )**2
                                ).unsqueeze(1).unsqueeze(1).unsqueeze(1)) \
                            * this_mask_sub_layer
                        self.square_avg['cnn'][block_idx][ec_layer_idx] = tmp_square_avg_blue + tmp_square_avg_red
                    elif self.args.rmsprop_method == 'noweight':

                        tmp_square_avg_blue =   \
                            (self.alpha *
                                self.square_avg['cnn'][block_idx][ec_layer_idx]
                            + (1 - self.alpha) *
                                (dw_blue)**2) \
                            * (1 - this_mask_sub_layer)
                        tmp_square_avg_red = \
                            (self.alpha *
                                self.square_avg['cnn'][block_idx][ec_layer_idx]
                            + (1 - self.alpha) *
                                ((( w_red*dw_red   - sigmadwd) /  (v_value) ) **2 ).unsqueeze(1).unsqueeze(1).unsqueeze(1)) \
                            * this_mask_sub_layer
                        self.square_avg['cnn'][block_idx][ec_layer_idx] = tmp_square_avg_blue + tmp_square_avg_red

            R = self.cal_R_cnn(
                                lr=lr_t,
                                w_red=w_red,
                                dw_red=dw_red,
                                sigmadwd=sigmadwd,
                                v_value=v_value,
                                w_norm=w_norm[this_block_s_idx],
                                mask=this_block_mask[this_block_s_idx],
                                square_avg= self.square_avg['cnn'][block_idx][this_block_s_idx]
                                )

            for ec_layer_idx, layer_idx in enumerate(block):
                this_mask = this_block_mask[ec_layer_idx]
                layer = self.params[layer_idx]
                out_shape, in_shape, shape_2, shape_3 = layer.data.shape

                this_mask_sub_layer_red = (this_mask == 1).to(torch.float32)
                this_mask_sub_layer_blue = (this_mask == 0).to(torch.float32)
                # print(lr_t,'!!!!!!!')

                if self.args.rmsprop_method == "weight":

                    lr = lr_t / (torch.sqrt(self.square_avg['cnn'][block_idx][ec_layer_idx]) + self.eps)
                elif self.args.rmsprop_method == "noweight":
                    lr = lr_t / ((torch.sqrt(self.square_avg['cnn'][block_idx][ec_layer_idx]) * w_norm[ec_layer_idx])+ self.eps)


                if ec_layer_idx == this_block_s_idx :
                    layer_is_s = True
                else:
                    layer_is_s = False

                if layer_is_s:
                    this_R_value = (R).unsqueeze(1).unsqueeze(1).unsqueeze(1)

                    tmp_red = (
                                      layer.data * this_R_value
                              ) * this_mask_sub_layer_red

                    tmp_blue_first = (
                                             layer.data - lr * layer.grad.data * w_norm[ec_layer_idx]
                                     ) * this_mask_sub_layer_blue

                    layer.data = tmp_blue_first + tmp_red


                elif ec_layer_idx > this_block_s_idx:
                    tmp_blue =  (layer.data -
                                 lr * layer.grad.data * w_norm[ec_layer_idx]
                                 /  (v_value[:in_shape] ** 2).unsqueeze(1).unsqueeze(1).unsqueeze(0)
                                 ) \
                                /(R[:in_shape]).unsqueeze(1).unsqueeze(1).unsqueeze(0) * this_mask_sub_layer_blue

                    tmp_red = layer.data * this_mask_sub_layer_red

                    layer.data = tmp_blue + tmp_red

                elif ec_layer_idx < this_block_s_idx:
                    tmp_blue =  (layer.data -
                                 lr * layer.grad.data  * w_norm[ec_layer_idx]
                                 /  (v_value[:out_shape] ** 2).unsqueeze(1).unsqueeze(1).unsqueeze(1)
                                 )  \
                                 /(R[:out_shape]).unsqueeze(1).unsqueeze(1).unsqueeze(1) * this_mask_sub_layer_blue

                    tmp_red = layer.data * this_mask_sub_layer_red

                    layer.data = tmp_blue + tmp_red



    def ec_step_mlp(self, blocks_idx, closure=None):
        """Performs a single optimization step."""
        lr_t = self.lr_t
        # if self.args.inner_opt != "sgd":
        #     lr = 1

            # print(lr,self.args.inner_opt)

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


            ## start calculate square_avg  buffer
            for ec_layer_idx, layer_idx in enumerate(block):
                layer = self.params[layer_idx]
                out_shape, in_shape = layer.data.shape

                this_mask_sub_layer = this_block_mask[ec_layer_idx]

                if ec_layer_idx != this_block_s_idx:
                    dw_blue = layer.grad.data * (1 - this_mask_sub_layer)

                    if ec_layer_idx < this_block_s_idx:

                        self.square_avg['mlp'][block_idx][ec_layer_idx] = \
                            (self.alpha *
                                    self.square_avg['mlp'][block_idx][ec_layer_idx]
                            + (1 - self.alpha) *
                                    (dw_blue / v_value[:out_shape].unsqueeze(1))**2
                             )*(1 - this_mask_sub_layer)      + this_mask_sub_layer

                    elif ec_layer_idx > this_block_s_idx:

                        self.square_avg['mlp'][block_idx][ec_layer_idx] = \
                            (self.alpha *
                                self.square_avg['mlp'][block_idx][ec_layer_idx]
                            + (1 - self.alpha) *
                                (dw_blue / v_value[:in_shape].unsqueeze(0))**2) \
                            *(1 - this_mask_sub_layer)    + this_mask_sub_layer

                elif ec_layer_idx == this_block_s_idx:
                    #  speical layer
                    # w_red = self.remain_output_mlp(layer.data * this_mask_sub_layer)
                    # dw_red = self.remain_output_mlp(layer.grad.data * this_mask_sub_layer)
                    dw_blue = layer.grad.data * (1 - this_mask_sub_layer)

                    tmp_square_avg_blue =   \
                        (self.alpha *
                            self.square_avg['mlp'][block_idx][ec_layer_idx]
                        + (1 - self.alpha) *
                            (dw_blue)**2) \
                        * (1 - this_mask_sub_layer)
                    tmp_square_avg_red = \
                        (self.alpha *
                            self.square_avg['mlp'][block_idx][ec_layer_idx]
                        + (1 - self.alpha) *
                            ((( w_red*dw_red   - sigmadwd) /  (v_value) ) **2 ).unsqueeze(1)) \
                        * this_mask_sub_layer
                    self.square_avg['mlp'][block_idx][ec_layer_idx] = tmp_square_avg_blue + tmp_square_avg_red




            R = self.cal_R_mlp(lr=lr_t,
                            w_red=w_red,
                            dw_red=dw_red,
                            sigmadwd=sigmadwd,
                            v_value=v_value,
                            mask=this_block_mask[this_block_s_idx],
                            square_avg=self.square_avg['mlp'][block_idx][this_block_s_idx]
                           )
            for ec_layer_idx, layer_idx in enumerate(block):
                this_mask = this_block_mask[ec_layer_idx]
                layer = self.params[layer_idx]
                lr = lr_t / (torch.sqrt(self.square_avg['mlp'][block_idx][ec_layer_idx]) + self.eps)
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
                                             layer.data - lr * layer.grad.data
                                     ) * this_mask_sub_layer_blue

                    layer.data = tmp_blue_first + tmp_red
                    # layer.data = (layer.data - lr * layer.grad.data) * (1 - this_mask) + \
                                 # layer.data * this_R_value * this_mask

                elif ec_layer_idx > this_block_s_idx:
                    out_shape, in_shape = layer.data.shape

                    tmp_blue =  (layer.data -
                                 lr * layer.grad.data
                                 /  (v_value[:in_shape] ** 2).unsqueeze(0)
                                 ) \
                                /(R[:in_shape]).unsqueeze(0) * this_mask_sub_layer_blue

                    tmp_red = layer.data * this_mask_sub_layer_red

                    layer.data = tmp_blue + tmp_red


                elif ec_layer_idx < this_block_s_idx:
                    out_shape, in_shape = layer.data.shape

                    tmp_blue =  (layer.data -
                                 lr * layer.grad.data
                                 /  (v_value[:out_shape] ** 2).unsqueeze(1)
                                 )  \
                                 /(R[:out_shape]).unsqueeze(1)  * this_mask_sub_layer_blue

                    tmp_red = layer.data * this_mask_sub_layer_red

                    layer.data = tmp_blue + tmp_red
