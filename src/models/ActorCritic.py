import torch.nn as nn
import torch.nn.functional as F

# Actor class for the agent, which will be used to take actions based on the current state.
class Actor(nn.Module):
    
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_size, 256)
        self.layer2 = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    
# Critic class for the agent, which will be used to evaluate the current state.
class Critic(nn.Module):

    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_size, 256)
        self.layer2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        value = self.layer2(x)
        return value