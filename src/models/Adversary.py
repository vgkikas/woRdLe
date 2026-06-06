# Teacher class for the agent, which will be used to take actions based on the current state.
import numpy as np
from collections import defaultdict

class Adversary:
    def __init__(self, vocab_path, alpha=0.01, epsilon=0.1):
        with open(vocab_path, 'r') as f:
            self.vocab = f.readlines()
        self.alpha = alpha
        self.epsilon = epsilon
        self.vocab_size = len(self.vocab)
        self.Q = defaultdict(lambda: 2 * np.ones(self.vocab_size))  #
        self.visit_count = defaultdict(int)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.vocab_size)
        else:
            return np.argwhere(self.Q['start'] == np.max(self.Q['start'])).flatten()[0]

    def observe(self, action, reward):
        self.visit_count[action] += 1
        self.Q['start'][action] += self.alpha * (reward + 110/self.visit_count[action] - self.Q['start'][action])
