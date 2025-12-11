#!/usr/bin/env python3

# Credits to: https://arxiv.org/pdf/1707.06347

# Would be quite funny if I did this whole thing in numpy
import numpy as np

def xavier_init(in_dim, out_dim):
    W = np.random.randn(in_dim, out_dim) * np.sqrt(2/out_dim)
    b = np.zeros(out_dim)
    return W, b

class Policy:
    def __init__(self, state_dim, action_dim, hidden=64):
        # Xavier Init for now
        self.W1, self.b1 = xavier_init(state_dim, hidden)
        self.W2, self.b2 = xavier_init(hidden, hidden)
        self.W3, self.b3 = xavier_init(hidden, action_dim)

    def forward(self, state):
        h1 = np.tanh(state @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3
        
