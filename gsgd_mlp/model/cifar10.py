import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Cifar10Net(nn.Module):
    def __init__(self, structure=(64,64,64,64,64)):
        super(Cifar10Net, self).__init__()
        self.structure = structure

        self.layer_list = torch.nn.ModuleList()

        self.layer_list.append(self.mack_layer(3072, structure[0] ,activation='relu'))
        for ii in range(len(self.structure) - 1):
            self.layer_list.append(
                self.mack_layer(self.structure[ii] , self.structure[ii+1],activation='relu')
            )
        self.layer_list.append(self.mack_layer(self.structure[-1] , 10 , activation=None))



    def mack_layer(self,input, output, activation='relu'):
        linear = nn.Linear(input, output, bias=False)
        relu = nn.ReLU()
        if activation == 'relu':
            return(nn.Sequential(linear, relu))
        else:
            return(nn.Sequential(linear))



    def forward(self, x):
        x = x.view(-1 , 3072 )
        for jj in self.layer_list:
            x = jj(x)

        return x
