import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class PathNet(nn.Module):
    def __init__(self, input_size = 28*28, hidden_size = 64, layer = 5):
        super(PathNet, self).__init__()
        self.input_size = input_size
        if layer < 3:
            raise Exception('layer number must greater than 2.')
        layers = [nn.Linear(input_size, hidden_size, bias=False), nn.ReLU()]
        for l in range(layer - 2):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 10, bias=False))
        self.linears = nn.Sequential(*layers)
        self.ec_layer = list(range(0, 3))


    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.linears(x)
        return x