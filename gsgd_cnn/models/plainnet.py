import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from.model_utils import plainnet_ec_layer

class PlainBlock(nn.Module):
	def __init__(self, in_planes, planes, stride=1):
		super(PlainBlock, self).__init__()
		self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn = nn.BatchNorm2d(planes, affine=True)

	def forward(self, x):
		out = F.relu(self.bn(self.conv(x)))
		return out

class PlainNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10):
		super(PlainNet, self).__init__()
		self.in_planes = 64

		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
		self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
		self.linear = nn.Linear(512, num_classes)

		self.layers = 2 + num_blocks[0] + num_blocks[1] + num_blocks[2] + num_blocks[3]

		self.ec_layer = {
			'cnn' : plainnet_ec_layer(num_blocks),
			'mlp' : []
		}

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes
		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out


def PlainNet34():
	return PlainNet(PlainBlock, [6,8,12,6])

def PlainNet34_C100():
	return PlainNet(PlainBlock, [6,8,12,6], num_classes=100)


def PlainNet110():
	return PlainNet(PlainBlock, [6,16,80,6])

def PlainNet110_C100():
	return PlainNet(PlainBlock, [6,16,80,6], num_classes=100)

def PlainNet6():
	return PlainNet(PlainBlock, [1,1,1,1])

def PlainNet10():
	return PlainNet(PlainBlock, [2,2,2,2])

def PlainNet14():
	return PlainNet(PlainBlock, [3,3,3,3])

class ACBlock(nn.Module):
	def __init__(self, in_planes, planes, stride=1):
		super(ACBlock, self).__init__()
		self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

	def forward(self, x):
		out = F.relu(self.conv(x))
		return out

class ACNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10):
		super(ACNet, self).__init__()
		self.in_planes = 64

		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
		self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
		self.linear = nn.Linear(512, num_classes)

		self.layers = 2 + num_blocks[0] + num_blocks[1] + num_blocks[2] + num_blocks[3]

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes
		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.conv1(x))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out


def ACNet10():
	return ACNet(ACBlock, [2,2,2,2])

#每个block最后一层做BN
class AC2Block(nn.Module):
	def __init__(self, in_planes, planes, stride=1):
		super(AC2Block, self).__init__()
		self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn = nn.BatchNorm2d(planes)

	def forward(self, x):
		out = F.relu(self.bn(self.conv(x)))
		return out


class ACNet2(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10):
		super(ACNet2, self).__init__()
		self.in_planes = 64

		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.layer1 = self._make_layer(block, 64, num_blocks[0]-1, stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[1]-1, stride=2)
		self.layer3 = self._make_layer(block, 256, num_blocks[2]-1, stride=2)
		self.layer4 = self._make_layer(block, 512, num_blocks[3]-1, stride=2)
		self.linear = nn.Linear(512, num_classes)

		self.layers = 2 + num_blocks[0] + num_blocks[1] + num_blocks[2] + num_blocks[3]

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes
		layers.append(AC2Block(self.in_planes, planes, stride=1))
		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.conv1(x))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out


def ACNet2_10():
	return ACNet2(ACBlock, [2,2,2,2])

def ACNet2_34():
	return ACNet2(ACBlock, [6,8,12,6])


class PlainBlock_UnAct(nn.Module):
	def __init__(self, in_planes, planes, stride=1):
		super(PlainBlock_UnAct, self).__init__()
		self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn = nn.BatchNorm2d(planes, affine=True)

	def forward(self, x):
		out = self.bn(self.conv(x))
		return out

class PlainNetB(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10):
		super(PlainNetB, self).__init__()
		self.in_planes = 64

		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
		self.shortcut2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128)
            )
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
		self.shortcut3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256)
            )
		self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
		self.shortcut4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512)
            )
		self.linear = nn.Linear(512, num_classes)

		self.layers = 2 + num_blocks[0] + num_blocks[1] + num_blocks[2] + num_blocks[3]

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides[0:(len(strides)-1)]:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes
		layers.append(PlainBlock_UnAct(self.in_planes, planes,strides[-1]))
		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.layer1(out) + out)
		out = F.relu(self.layer2(out) + self.shortcut2(out))
		out = F.relu(self.layer3(out) + self.shortcut3(out))
		out = F.relu(self.layer4(out) + self.shortcut4(out))
		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out


def PlainNetB34():
	return PlainNetB(PlainBlock, [6,8,12,6])

def PlainNetB6():
	return PlainNetB(PlainBlock, [1,1,1,1])

def PlainNetB10():
	return PlainNetB(PlainBlock, [2,2,2,2])

def PlainNetB14():
	return PlainNetB(PlainBlock, [3,3,3,3])