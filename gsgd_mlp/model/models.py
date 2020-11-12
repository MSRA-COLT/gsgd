import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 1, bias=False)
        self.fc2 = nn.Linear(1, 2, bias=False)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x



class SimpleNet_3333(nn.Module):
    def __init__(self):
        super(SimpleNet_3333, self).__init__()
        self.fc1 = nn.Linear(3, 3, bias=False)
        self.fc2 = nn.Linear(3, 3, bias=False)
        self.fc3 = nn.Linear(3, 3, bias=False)



    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x



class SimpleNet_43432(nn.Module):
    def __init__(self):
        super(SimpleNet_43432, self).__init__()
        self.fc1 = nn.Linear(4, 3, bias=False)
        self.fc2 = nn.Linear(3, 4, bias=False)
        self.fc3 = nn.Linear(4, 3, bias=False)
        self.fc4 = nn.Linear(3, 2, bias=False)



    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x




class SimpleNet_35362(nn.Module):
    def __init__(self):
        super(SimpleNet_35362, self).__init__()
        self.fc1 = nn.Linear(3, 5, bias=False)
        self.fc2 = nn.Linear(5, 3, bias=False)
        self.fc3 = nn.Linear(3, 6, bias=False)
        self.fc4 = nn.Linear(6, 2, bias=False)



    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x

class SimpleNet_den(nn.Module):
    def __init__(self,structure=( 4,2,4,2,4,2,4 ),input_size=3,output_size=3):
        ### 3 4 2 4 2 4 2 4 3
        super(SimpleNet_den, self).__init__()
        self.structure = structure
        self.input_size= input_size
        self.output_size = output_size
        self.ec_layer = list(range(0,len(structure)+1))
        self.layer_list = torch.nn.ModuleList()
        self.layer_list.append(self.mack_layer(self.input_size, structure[0] ,activation='relu',bias=False))
        for ii in range(len(self.structure) - 1):
            self.layer_list.append(
                self.mack_layer(self.structure[ii] , self.structure[ii+1],activation='relu')
            )
        self.layer_list.append(self.mack_layer(self.structure[-1] , self.output_size , activation=None,bias=False))

    def mack_layer(self,input, output, activation='relu', bias=False):
        linear = nn.Linear(input, output, bias=bias)
        relu = nn.ReLU()
        if activation == 'relu':
            return(nn.Sequential(linear, relu))
        else:
            return(nn.Sequential(linear))



    def forward(self, x):
        for jj in self.layer_list:
            x = jj(x)

        return x

class SimpleNet_den_mnist(nn.Module):
    def __init__(self, structure=(  128, 64, 128,  64, 128), input_size=28*28, output_size=10):
        ### 3 4 2 4 2 4 2 4 3
        super(SimpleNet_den_mnist, self).__init__()
        self.structure = structure
        self.input_size = input_size
        self.output_size = output_size
        self.ec_layer = list(range(0, len(structure) +1))
        self.layer_list = torch.nn.ModuleList()
        self.layer_list.append(self.mack_layer(self.input_size, structure[0], activation='relu', bias=False))
        for ii in range(len(self.structure) - 1):
            self.layer_list.append(
                self.mack_layer(self.structure[ii], self.structure[ii + 1], activation='relu')
            )
        self.layer_list.append(self.mack_layer(self.structure[-1], self.output_size, activation=None, bias=False))

    def mack_layer(self, input, output, activation='relu', bias=False):
        linear = nn.Linear(input, output, bias=bias)
        relu = nn.ReLU()
        if activation == 'relu':
            return (nn.Sequential(linear, relu))
        else:
            return (nn.Sequential(linear))

    def forward(self, x):
        x = x.view(-1, 28*28)
        for jj in self.layer_list:
            x = jj(x)

        return x


class SimpleNet_den_cifar(nn.Module):
    def __init__(self, structure=( 128, 64, 128,  64,128), input_size=3072, output_size=10):
        ### 3 4 2 4 2 4 2 4 3
        super(SimpleNet_den_cifar, self).__init__()
        self.structure = structure
        self.input_size = input_size
        self.output_size = output_size
        self.ec_layer = list(range(0, len(structure)+1 ))
        self.layer_list = torch.nn.ModuleList()
        self.layer_list.append(self.mack_layer(self.input_size, structure[0], activation='relu', bias=False))
        for ii in range(len(self.structure) - 1):
            self.layer_list.append(
                self.mack_layer(self.structure[ii], self.structure[ii + 1], activation='relu')
            )
        self.layer_list.append(self.mack_layer(self.structure[-1], self.output_size, activation=None, bias=False))

    def mack_layer(self, input, output, activation='relu', bias=False):
        linear = nn.Linear(input, output, bias=bias)
        relu = nn.ReLU()
        if activation == 'relu':
            return (nn.Sequential(linear, relu))
        else:
            return (nn.Sequential(linear))

    def forward(self, x):
        x = x.view(-1, 3072)
        for jj in self.layer_list:
            x = jj(x)

        return x


class SimpleNet_den_mnist_2(nn.Module):
    def __init__(self, structure=(  64, 32, 64,  32, 64), input_size=28*28, output_size=10):
        ### 3 4 2 4 2 4 2 4 3
        super(SimpleNet_den_mnist_2, self).__init__()
        self.structure = structure
        self.input_size = input_size
        self.output_size = output_size
        self.ec_layer = list(range(0, len(structure) +1))
        self.layer_list = torch.nn.ModuleList()
        self.layer_list.append(self.mack_layer(self.input_size, structure[0], activation='relu', bias=False))
        for ii in range(len(self.structure) - 1):
            self.layer_list.append(
                self.mack_layer(self.structure[ii], self.structure[ii + 1], activation='relu')
            )
        self.layer_list.append(self.mack_layer(self.structure[-1], self.output_size, activation=None, bias=False))

    def mack_layer(self, input, output, activation='relu', bias=False):
        linear = nn.Linear(input, output, bias=bias)
        relu = nn.ReLU()
        if activation == 'relu':
            return (nn.Sequential(linear, relu))
        else:
            return (nn.Sequential(linear))

    def forward(self, x):
        x = x.view(-1, 28*28)
        for jj in self.layer_list:
            x = jj(x)

        return x


class SimpleNet_den_cifar_2(nn.Module):
    def __init__(self, structure=( 64, 32, 64,  32, 64), input_size=3072, output_size=10):
        ### 3 4 2 4 2 4 2 4 3
        super(SimpleNet_den_cifar_2, self).__init__()
        self.structure = structure
        self.input_size = input_size
        self.output_size = output_size
        self.ec_layer = list(range(0, len(structure)+1 ))
        self.layer_list = torch.nn.ModuleList()
        self.layer_list.append(self.mack_layer(self.input_size, structure[0], activation='relu', bias=False))
        for ii in range(len(self.structure) - 1):
            self.layer_list.append(
                self.mack_layer(self.structure[ii], self.structure[ii + 1], activation='relu')
            )
        self.layer_list.append(self.mack_layer(self.structure[-1], self.output_size, activation=None, bias=False))

    def mack_layer(self, input, output, activation='relu', bias=False):
        linear = nn.Linear(input, output, bias=bias)
        relu = nn.ReLU()
        if activation == 'relu':
            return (nn.Sequential(linear, relu))
        else:
            return (nn.Sequential(linear))

    def forward(self, x):
        x = x.view(-1, 3072)
        for jj in self.layer_list:
            x = jj(x)

        return x


class SimpleNet_den_mnist_3(nn.Module):
    def __init__(self, structure=(  96, 64, 96, 64, 96), input_size=28*28, output_size=10):
        ### 3 4 2 4 2 4 2 4 3
        super(SimpleNet_den_mnist_3, self).__init__()
        self.structure = structure
        self.input_size = input_size
        self.output_size = output_size
        self.ec_layer = list(range(0, len(structure) +1))
        self.layer_list = torch.nn.ModuleList()
        self.layer_list.append(self.mack_layer(self.input_size, structure[0], activation='relu', bias=False))
        for ii in range(len(self.structure) - 1):
            self.layer_list.append(
                self.mack_layer(self.structure[ii], self.structure[ii + 1], activation='relu')
            )
        self.layer_list.append(self.mack_layer(self.structure[-1], self.output_size, activation=None, bias=False))

    def mack_layer(self, input, output, activation='relu', bias=False):
        linear = nn.Linear(input, output, bias=bias)
        relu = nn.ReLU()
        if activation == 'relu':
            return (nn.Sequential(linear, relu))
        else:
            return (nn.Sequential(linear))

    def forward(self, x):
        x = x.view(-1, 28*28)
        for jj in self.layer_list:
            x = jj(x)

        return x


class SimpleNet_den_cifar_3(nn.Module):
    def __init__(self, structure=( 96, 64, 96, 64, 96), input_size=3072, output_size=10):
        ### 3 4 2 4 2 4 2 4 3
        super(SimpleNet_den_cifar_3, self).__init__()
        self.structure = structure
        self.input_size = input_size
        self.output_size = output_size
        self.ec_layer = list(range(0, len(structure)+1 ))
        self.layer_list = torch.nn.ModuleList()
        self.layer_list.append(self.mack_layer(self.input_size, structure[0], activation='relu', bias=False))
        for ii in range(len(self.structure) - 1):
            self.layer_list.append(
                self.mack_layer(self.structure[ii], self.structure[ii + 1], activation='relu')
            )
        self.layer_list.append(self.mack_layer(self.structure[-1], self.output_size, activation=None, bias=False))

    def mack_layer(self, input, output, activation='relu', bias=False):
        linear = nn.Linear(input, output, bias=bias)
        relu = nn.ReLU()
        if activation == 'relu':
            return (nn.Sequential(linear, relu))
        else:
            return (nn.Sequential(linear))

    def forward(self, x):
        x = x.view(-1, 3072)
        for jj in self.layer_list:
            x = jj(x)

        return x




class SimpleNet_48084(nn.Module):
    def __init__(self):
        super(SimpleNet_48084, self).__init__()
        self.fc1 = nn.Linear(4, 8, bias=False)
        self.fc2 = nn.Linear(8, 10, bias=False)
        self.fc3 = nn.Linear(10, 8, bias=False)
        self.fc4 = nn.Linear(8, 4, bias=False)



    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x


class SimpleNet_4488008844(nn.Module):
    def __init__(self):
        super(SimpleNet_4488008844, self).__init__()
        self.fc1 = nn.Linear(4, 4, bias=False)
        self.fc2 = nn.Linear(4, 8, bias=False)
        self.fc3 = nn.Linear(8, 8, bias=False)
        self.fc4 = nn.Linear(8, 10, bias=False)
        self.fc5 = nn.Linear(10, 10, bias=False)
        self.fc6 = nn.Linear(10, 8, bias=False)
        self.fc7 = nn.Linear(8, 8, bias=False)
        self.fc8 = nn.Linear(8, 4, bias=False)
        self.fc9 = nn.Linear(4, 4, bias=False)




    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        x = F.relu(x)
        x = self.fc9(x)

        return x



class SimpleNet_cifar10(nn.Module):
    def __init__(self):
        super(SimpleNet_cifar10, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 64, bias=False)
        self.fc2 = nn.Linear(64, 64, bias=False)
        self.fc3 = nn.Linear(64, 64, bias=False)
        self.fc4 = nn.Linear(64, 64, bias=False)
        self.fc5 = nn.Linear(64, 64, bias=False)
        self.fc6 = nn.Linear(64, 10, bias=False)


    def forward(self, x):
        x = x.view(-1 , 3*32*32 )
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        return x

class SimpleNet_mnist(nn.Module):
    def __init__(self):
        super(SimpleNet_mnist, self).__init__()
        self.fc1 = nn.Linear(28*28, 64, bias=False)
        self.fc2 = nn.Linear(64, 128, bias=False)
        # self.shortcut_1 = nn.Linear(28*28, 64, bias=False)
        self.fc3 = nn.Linear(128, 256, bias=False)
        self.fc4 = nn.Linear(256, 64, bias=False)
        # self.shortcut_2 = nn.Linear(64, 64, bias=False)
        self.fc5 = 	nn.Linear(64, 10, bias=False)

        # self.fc1 = nn.Linear(28 * 28, 64, bias=False)
        # self.fc2 = nn.Linear(64, 64, bias=False)
        # # self.shortcut_1 = nn.Linear(28*28, 64, bias=False)
        # self.fc3 = nn.Linear(64, 64, bias=False)
        # self.fc4 = nn.Linear(64, 64, bias=False)
        # # self.shortcut_2 = nn.Linear(64, 64, bias=False)
        # self.fc5 = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        x = x.view(-1 , 28*28 )
        o1 = self.fc1(x)
        o1 = F.relu(o1)
        o1 = self.fc2(o1)
        o2 = F.relu(o1)
        # o2 = F.relu(self.shortcut_1(x) + o1)
        o3 = self.fc3(o2)
        o3 = F.relu(o3)
        o3 = self.fc4(o3)
        o4 = F.relu(o3)
        # o4 = F.relu(self.shortcut_2(o2) + o3)
        x = self.fc5(o4)
        return x




class SimpleNet_detail(nn.Module):
    def __init__(self):
        super(SimpleNet_detail, self).__init__()
        self.param = nn.ParameterDict(
            {
                'a1' : nn.Parameter(torch.tensor(1.0)),
                'a2': nn.Parameter(torch.tensor(1.0)),
                'b1': nn.Parameter(torch.tensor(1.0)),
                'b2': nn.Parameter(torch.tensor(1.0)),
            }
        )


    def forward(self, x):
        x1 = x[:,:1]
        x2 = x[:,1:2]
        h1  = self.param['a1'] *x1 + self.param['a2'] * x2
        rh1 = F.relu(h1)
        o1 = rh1 * self.param['b1']
        o2 = rh1 * self.param['b2']
        # print(x1.shape , rh1.shape, o1.shape)
        return torch.cat((o1, o2),1)




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
