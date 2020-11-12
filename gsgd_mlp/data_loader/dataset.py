import torch.utils.data as data
import torch
from torchvision import datasets,transforms

import copy

class simulation(data.Dataset):
    def __init__(self, n_data, n_feature, model, noise_y=0,  seed = 1):
        torch.manual_seed(seed)
        self.n_data =n_data
        self.n_feature = n_feature
        self.model = model
        self.noise_y = noise_y
        self.generate_data(n_data)







    def  generate_data(self, num):
        with torch.no_grad():
            self.data = torch.randn(num , self.n_feature) / (len(self.model.state_dict()))
            # self.data = torch.abs(self.data)
            target  = self.model(self.data)
            # print(self.data, target)
            self.target  = target  + self.noise_y * torch.randn(target.shape)



    def __getitem__(self, index):
        x,y = self.data[index] , self.target[index]

        return x,y


    def __len__(self):
        return(self.n_data)

def simulation_data(n_data_train, n_data_test, n_feature , noise_y , noise_model , model,train_batch_size,test_batch_size ):
    model = copy.deepcopy(model)
    for ii in model.parameters():
        ii.data += torch.randn_like(ii.data) * noise_model

    train_loader = torch.utils.data.DataLoader(simulation(n_data =n_data_train,
                                                          n_feature=n_feature,
                                                          noise_y=noise_y,
                                                          model=model),batch_size=train_batch_size, shuffle=False,)
    test_loader = torch.utils.data.DataLoader(simulation(n_data =n_data_test,
                                                          n_feature=n_feature,
                                                          noise_y=noise_y,
                                                          model=model),batch_size=test_batch_size, shuffle=False,)
    return (train_loader,test_loader)



def mnist(batch_size,test_batch_size, kwargs = (1)):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True,  )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/mnist', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True,  )

    return (train_loader,test_loader)



def cifar10(batch_size,test_batch_size, kwargs = (1)):
    # kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    trainset = datasets.CIFAR10(root='../data/cifar10', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    testset = datasets.CIFAR10(root='../data/cifar10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=8)
    return (train_loader,test_loader)