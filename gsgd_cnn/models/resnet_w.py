'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def cal_identity(x):
    print(x.shape)
    x1, x2, x3, x4 = x.shape
    center = x4 // 2
    idx_tmp = list(zip(list(range(x1)), list(range(x2))))
    idx = torch.LongTensor([[*x, center, center] for x in idx_tmp]).transpose(0, 1)
    value = torch.ones(x1)
    shape = x.shape

    res = torch.sparse.FloatTensor(idx, value, shape).to_dense()
    return (res)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(self.expansion*planes)
        )
        if stride  == 1 or in_planes == self.expansion * planes:
            for ii in self.shortcut.parameters():
                ii.data = cal_identity(ii.data)
                break


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_w(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_w, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        if block.__name__ == 'BasicBlock':
            self.layers = 2 + 2 * (num_blocks[0] + num_blocks[1] + num_blocks[2] + num_blocks[3])
        elif block.__name__ == 'Bottleneck':
            self.layers = 2 + 3 * (num_blocks[0] + num_blocks[1] + num_blocks[2] + num_blocks[3])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
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


def ResNet_w18():
    return ResNet_w(BasicBlock, [2,2,2,2])

def ResNet_w18_C100():
    return ResNet_w(BasicBlock, [2,2,2,2],num_classes=100)

def ResNet_w34():
    return ResNet_w(BasicBlock, [3,4,6,3])

def ResNet_w34_C100():
    return ResNet_w(BasicBlock, [3,4,6,3],num_classes=100)

# def ResNet_w50():
#     return ResNet_w(Bottleneck, [3,4,8,3])

# def ResNet_w101():
#     return ResNet_w(Bottleneck, [3,4,23,3])
#
# def ResNet_w152():
#     return ResNet_w(Bottleneck, [3,8,36,3])
#
def ResNet_w110():
    return ResNet_w(BasicBlock, [3,8,40,3])

def ResNet_w110_C100():
    return ResNet_w(BasicBlock, [3,8,40,3], num_classes=100)


def ResNet_w110_userdef():
    res =  ResNet_w(BasicBlock, [3,8,40,3])
    conv_idx = [
                [[3, 6, 12, 15, 21, 24]],
                [[30, 33, 39, 42, 48, 51, 57,  60], [66, 69, 75, 78, 84, 87, 93, 96]],
                [[102, 105, 111, 114, 120, 123, 129, 132], [138, 141, 147, 150, 156, 159, 165, 168], [174, 177, 183, 186,
                  192, 195, 201, 204], [210, 213, 219, 222, 228, 231, 237, 240], [246, 249, 255, 258, 264, 267, 273, 276], [
                  282, 285, 291, 294, 300, 303], [309, 312, 318, 321, 327, 330, 336, 339], [345, 348, 354, 357, 363, 366,
                  372, 375], [381, 384, 390, 393, 399, 402, 408, 411], [417, 420, 426, 429, 435, 438, 444, 447, 453, 456]],
                [[462, 465, 471, 474, 480, 483]]

                ]
    shortcut_idx =[
                [[[[9], [3, 6]], [[18], [12, 15]], [[27], [21, 24]]]],
                [   [[[36], [30, 33]], [[45], [39, 42]], [[54], [48, 51]], [[63], [57, 60]]],
                    [[[72], [66, 69]], [[81], [75, 78]], [[90], [84, 87]], [[99], [93, 96]]]
                ],
                [   [[[108], [102, 105]], [[117], [111, 114]], [[126], [120, 123]], [[135], [129, 132]] ],
                    [[[144], [138, 141]], [[153], [147, 150]], [[162], [156, 159]], [[171], [165, 168]]],
                    [[[180], [174, 177]], [[189], [183, 186]], [[198], [192, 195]], [[207], [201, 204]]],
                    [[[216], [210, 213]], [[225], [219, 222]], [[234], [228, 231]], [[243], [237, 240]]],
                    [[[252], [246, 249]], [[261], [255, 258]], [[270], [264, 267]], [[279], [273, 276]]],
                    [[[288], [282, 285]], [[297], [291, 294]], [[306], [300, 303]], [[315], [309, 312]]],
                    [[[324], [318, 321]], [[333], [327, 330]], [[342], [336, 339]], [[351], [345, 348]]],
                    [[[360], [354, 357]], [[369], [363, 366]], [[378], [372, 375]], [[387], [381, 384]]],
                    [[[396], [390, 393]], [[405], [399, 402]], [[414], [408, 411]], [[423], [417, 420]]],
                    [[[432], [426, 429]], [[441], [435, 438]], [[450], [444, 447]], [[459], [453, 456]]]
                ],
                [[[[468], [462, 465]], [[477], [471, 474]], [[486], [480, 483]]]]]
    res.conv_idx =[conv_idx, shortcut_idx]
    return res

def ResNet_w110_C100_userdef():
    res =  ResNet_w(BasicBlock, [3,8,40,3], num_classes=100)
    conv_idx = [
                [[3, 6, 12, 15, 21, 24]],
                [[30, 33, 39, 42, 48, 51, 57,  60], [66, 69, 75, 78, 84, 87, 93, 96]],
                [[102, 105, 111, 114, 120, 123, 129, 132], [138, 141, 147, 150, 156, 159, 165, 168], [174, 177, 183, 186,
                  192, 195, 201, 204], [210, 213, 219, 222, 228, 231, 237, 240], [246, 249, 255, 258, 264, 267, 273, 276], [
                  282, 285, 291, 294, 300, 303], [309, 312, 318, 321, 327, 330, 336, 339], [345, 348, 354, 357, 363, 366,
                  372, 375], [381, 384, 390, 393, 399, 402, 408, 411], [417, 420, 426, 429, 435, 438, 444, 447, 453, 456]],
                [[462, 465, 471, 474, 480, 483]]

                ]
    shortcut_idx =[
                [[[[9], [3, 6]], [[18], [12, 15]], [[27], [21, 24]]]],
                [   [[[36], [30, 33]], [[45], [39, 42]], [[54], [48, 51]], [[63], [57, 60]]],
                    [[[72], [66, 69]], [[81], [75, 78]], [[90], [84, 87]], [[99], [93, 96]]]
                ],
                [   [[[108], [102, 105]], [[117], [111, 114]], [[126], [120, 123]], [[135], [129, 132]] ],
                    [[[144], [138, 141]], [[153], [147, 150]], [[162], [156, 159]], [[171], [165, 168]]],
                    [[[180], [174, 177]], [[189], [183, 186]], [[198], [192, 195]], [[207], [201, 204]]],
                    [[[216], [210, 213]], [[225], [219, 222]], [[234], [228, 231]], [[243], [237, 240]]],
                    [[[252], [246, 249]], [[261], [255, 258]], [[270], [264, 267]], [[279], [273, 276]]],
                    [[[288], [282, 285]], [[297], [291, 294]], [[306], [300, 303]], [[315], [309, 312]]],
                    [[[324], [318, 321]], [[333], [327, 330]], [[342], [336, 339]], [[351], [345, 348]]],
                    [[[360], [354, 357]], [[369], [363, 366]], [[378], [372, 375]], [[387], [381, 384]]],
                    [[[396], [390, 393]], [[405], [399, 402]], [[414], [408, 411]], [[423], [417, 420]]],
                    [[[432], [426, 429]], [[441], [435, 438]], [[450], [444, 447]], [[459], [453, 456]]]
                ],
                [[[[468], [462, 465]], [[477], [471, 474]], [[486], [480, 483]]]]]
    res.conv_idx =[conv_idx, shortcut_idx]
    return  res


def ResNet_w34_userdef():
    res =  ResNet_w(BasicBlock, [3,4,6,3])
    conv_idx = [
                    [[3, 6, 12, 15, 21, 24]],
                    [[30, 33, 39, 42, 48, 51, 57, 60]],
                    [[66, 69, 75, 78, 84, 87],
                        [93, 96, 102, 105, 111, 114]],
                    [[120, 123, 129, 132, 138, 141]]
                ]
    shortcut_idx = [
                        [[[[9], [3, 6]], [[18], [12, 15]], [[27], [21, 24]]]],
                        [[[[36], [30, 33]], [[45], [39, 42]], [[54], [48, 51]], [[63], [57, 60]]]],
                        [   [[[72], [66, 69]], [[81], [75, 78]], [[90], [84, 87]]],
                            [[[99], [93, 96]], [[108], [102, 105]], [[117], [111, 114]]]
                        ],
                        [[[[126], [120, 123]], [[135], [129, 132]], [[144], [138, 141]]]]
                    ]

    res.conv_idx =[conv_idx, shortcut_idx]
    return res

def ResNet_w34_C100_userdef():
    res =  ResNet_w(BasicBlock, [3,4,6,3],num_classes=100)
    conv_idx = [
                    [[3, 6, 12, 15, 21, 24]],
                    [[30, 33, 39, 42, 48, 51, 57, 60]],
                    [[66, 69, 75, 78, 84, 87],
                        [93, 96, 102, 105, 111, 114]],
                    [[120, 123, 129, 132, 138, 141]]
                ]
    shortcut_idx = [
                        [[[[9], [3, 6]], [[18], [12, 15]], [[27], [21, 24]]]],
                        [[[[36], [30, 33]], [[45], [39, 42]], [[54], [48, 51]], [[63], [57, 60]]]],
                        [   [[[72], [66, 69]], [[81], [75, 78]], [[90], [84, 87]]],
                            [[[99], [93, 96]], [[108], [102, 105]], [[117], [111, 114]]]
                        ],
                        [[[[126], [120, 123]], [[135], [129, 132]], [[144], [138, 141]]]]
                    ]

    res.conv_idx =[conv_idx, shortcut_idx]
    return res




# def test():
#     net = ResNet_w18()
#     y = net(Variable(torch.randn(1,3,32,32)))
#     print(y.size())
#
# # test()
