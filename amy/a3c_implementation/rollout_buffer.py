import torch
import numpy as np 

# pulled from Hulusi's branch:
class RolloutBuffer:
    '''
    Branching slightly off of the papers implementation, fundementally I believe the buffer is purely meant to store and retrieve.
    Which is why compute_gae() is a standalone and get() returns the full buffer.
    '''
    def __init__(self):
        self.clear()
    
    # Aims to append a single timestep data
    def store(self, obs, action, reward, log_prob, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    # Reset for next rollout
    def clear(self):
        self.obs, self.actions, self.rewards = [], [], []
        self.log_probs, self.values, self.dones = [], [], []

    
    # Algorithm 1 (Page 5): "optimize it with minibatch SGD... for K epochs."
    # Which means we would need to shuffle and yield mini batches
    # As of now get() returns the full buffer, which means ill handle batching in the training loop instead 
    def get(self):
        return (
            torch.tensor(np.array(self.obs), dtype=torch.float32),
            torch.tensor(self.actions, dtype=torch.long),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.log_probs, dtype=torch.float32),
            torch.tensor(self.values, dtype=torch.float32),
            torch.tensor(self.dones, dtype=torch.float32)
        )