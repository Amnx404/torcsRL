import torch
import torch.nn as nn
import torch.optim as optim

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 300)
        self.fc2 = nn.Linear(300, 600)
        self.steering = nn.Linear(600, 1)
        self.acceleration = nn.Linear(600, 1)
        self.brake = nn.Linear(600, 1)
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        steering = self.tanh(self.steering(x))
        acceleration = self.sigmoid(self.acceleration(x))
        brake = self.sigmoid(self.brake(x))
        return torch.cat((steering, acceleration, brake), 1)

