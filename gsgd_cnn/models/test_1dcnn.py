import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math

class Conv1D_chl_3_multi(nn.Module):
    def __init__(self, input_chl, input_dim, kernel_size, fc_list, chl_list, dp, pool_type):
        #
        super(Conv1D_chl_3_multi, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc_list[0])
        self.fc2 = nn.Linear(fc_list[0], fc_list[1])

        self.conv1 = nn.Conv1d(input_chl, chl_list[0], kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(chl_list[0], chl_list[1], kernel_size=kernel_size)
        self.fc3 = nn.Linear(chl_list[1] + fc_list[1], fc_list[2])
        self.dp = nn.Dropout(dp)
        if pool_type == 'Avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = nn.AdaptiveMaxPool1d(1)

        self.ec_layer = {
            'cnn': [[4, 6]],
            'mlp': [[0, 2]],
        }


    def forward(self, x1, x2):
        x1 = self.dp(F.relu(self.conv1(x1)))
        x1 = self.dp(F.relu(self.conv2(x1)))
        x1 = self.pool(x1).squeeze(-1)

        x2 = self.dp(F.relu(self.fc1(x2)))
        x2 = self.dp(F.relu(self.fc2(x2)))

        return self.fc3(torch.cat([x1, x2], dim=1))
