import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='relu', bias=False, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes, bias=False)
    
    def forward(self, x):
        # Set initial states 
        # h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()) 
        # c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)  
        
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])  
        return out

def RNN_Mnist():
	return RNN(28, 128, 4, 10)