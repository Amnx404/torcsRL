import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(CriticNetwork, self).__init__()
        self.fcs1 = nn.Linear(state_size, 300)
        self.fcs2 = nn.Linear(300, 600)
        self.fca1 = nn.Linear(action_size, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, action_size)

    def forward(self, state, action):
        s = torch.relu(self.fcs1(state))
        s = self.fcs2(s)
        a = self.fca1(action)
        x = torch.relu(s + a)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

