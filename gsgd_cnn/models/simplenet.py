import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math

def initweight(layer, isnormal = False):
    if isinstance(layer, nn.Linear):
        stdv = 1. / math.sqrt(layer.weight.size(1))
        if not isnormal:
            layer.weight.data.uniform_(-stdv, stdv)
        else:
            layer.weight.data.normal_(0, stdv)
    elif isinstance(layer, nn.Conv2d):
        n = layer.in_channels
        for k in layer.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        if not isnormal:
            layer.weight.data.uniform_(-stdv, stdv)
        else:
            layer.weight.data.normal_(0, stdv)
    layer.weight.data.abs_()

class MNIST_Net(nn.Module):
	def __init__(self, bias = False):
		super(MNIST_Net, self).__init__()
		self.fc1 = nn.Linear(28*28, 64, bias=bias)
		self.fc3 = nn.Linear(64, 64, bias=bias)
		self.fc4 = nn.Linear(64, 64, bias=bias)
		self.fc5 = nn.Linear(64, 64, bias=bias)
		self.fc2 = nn.Linear(64, 10, bias=bias)

	def forward(self, x):
		x = x.view(-1, 28*28)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = F.relu(self.fc5(x))
		x = self.fc2(x)
		return x

class CIFAR_Net(nn.Module):
	def __init__(self, bias = False):
		super(CIFAR_Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, 5, bias = bias)
		self.conv2 = nn.Conv2d(64, 64, 5, bias = bias)
		self.fc1 = nn.Linear(64*5*5, 256, bias = bias)
		self.fc2 = nn.Linear(256, 256, bias = bias)
		self.fc3 = nn.Linear(256, 10, bias = bias)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2)
		x = x.view(-1, 64*5*5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return x

class CIFAR_BNNet(nn.Module):
	def __init__(self, bias = False):
		super(CIFAR_BNNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, 5, bias = bias)
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 64, 5, bias = bias)
		self.bn2 = nn.BatchNorm2d(64)
		self.fc1 = nn.Linear(64*5*5, 256, bias = bias)
		self.fc2 = nn.Linear(256, 256, bias = bias)
		self.fc3 = nn.Linear(256, 10, bias = bias)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.max_pool2d(x, 2)
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.max_pool2d(x, 2)
		x = x.view(-1, 64*5*5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return x

class PathNet_Mnist(nn.Module):
	def __init__(self, bias = False):
		super(PathNet_Mnist, self).__init__()
		self.fc1 = nn.Linear(28*28, 4000, bias=bias)
		self.fc3 = nn.Linear(4000, 4000, bias=bias)
		self.fc2 = nn.Linear(4000, 10, bias=bias)

	def forward(self, x):
		x = x.view(-1, 28*28)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc3(x))
		x = self.fc2(x)
		return x

class PathNet_Cifar(nn.Module):
	def __init__(self, bias = False):
		super(PathNet_Cifar, self).__init__()
		self.fc1 = nn.Linear(32*32*3, 4000, bias=bias)
		self.fc3 = nn.Linear(4000, 4000, bias=bias)
		self.fc2 = nn.Linear(4000, 10, bias=bias)

	def forward(self, x):
		x = x.view(-1, 32*32*3)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc3(x))
		x = self.fc2(x)
		return x

class PathNet(nn.Module):
	def __init__(self, input_size = 28*28, hidden_size = 4000, layer = 3):
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


	def forward(self, x):
		x = x.view(-1, self.input_size)
		x = self.linears(x)
		return x
		