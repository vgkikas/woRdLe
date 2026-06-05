import torch.nn as nn
from torch.nn.functional import relu

# Actor class for the agent, which will be used to take actions based on the current state.
class Actor(nn.Module):
    
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_size, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, action_size)

    def forward(self, x):
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = relu(self.layer3(x))
        x = self.layer4(x)
        return x
    
# Critic class for the agent, which will be used to evaluate the current state.
class Critic(nn.Module):

    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_size, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 1)
    def forward(self, x):
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = relu(self.layer3(x))
        value = self.layer4(x)
        return value