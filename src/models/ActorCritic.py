import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.distributions import Categorical

# Actor class for the agent, which will be used to take actions based on the current state.
class Actor(nn.Module):
    def __init__(self, state_size, action_size, ohe_matrix):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_size, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, action_size)

        self.register_buffer('ohe_matrix', torch.tensor(ohe_matrix, dtype=torch.float32))

    def forward(self, x):
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = relu(self.layer3(x))
        x = self.layer4(x)
        return x

    def get_distribution(self, output):
        """
        Takes the 130-d actor output, multiplies with each and converts it to actual logits
        and a Categorical distribution over the vocabulary.
        """
        # Inner product of actor output and one-hot encoded word, analogous to cosine similarity
        logits = torch.matmul(output, self.ohe_matrix)
        return Categorical(logits=logits)
    
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