import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.distributions import Categorical

# Actor class for the agent, which will be used to take actions based on the current state.
class Actor(nn.Module):
    def __init__(self, state_size, ohe_matrix):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_size, 256)
        self.layer2 = nn.Linear(256, ohe_matrix.shape[0])

        self.register_buffer('ohe_matrix', torch.tensor(ohe_matrix, dtype=torch.float32))

    def forward(self, x):
        x = relu(self.layer1(x))
        return self.layer2(x)

    def get_distribution(self, output, mask=None):
        """
        Takes the 130-d actor output, computes the inner product with each word in the vocabulary
        and passes the output to a categorical distribution.
        """
        # Inner product of actor output and one-hot encoded word, analogous to cosine similarity
        logits = torch.matmul(output, self.ohe_matrix)
        if mask is not None:
            logits += mask
        return Categorical(logits=logits)
    
# Critic class for the agent, which will be used to evaluate the current state.
class Critic(nn.Module):

    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.layer = nn.Linear(state_size, 1)
        # self.layer1 = nn.Linear(state_size, 256)
        # self.layer2 = nn.Linear(256, 256)
        # self.layer3 = nn.Linear(256, 256)
        # self.layer4 = nn.Linear(256, ohe_matrix.shape[0])
    def forward(self, x):
        # x = relu(self.layer1(x))
        # # x = relu(self.layer2(x))
        # # x = relu(self.layer3(x))
        # value = self.layer4(x)
        return self.layer(x)