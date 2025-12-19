#!/usr/bin/env python3

# Credits:  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
#           https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
#
# Inspiration: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_normalize.py

# Core idea is:
# Keep running statistics of observations. Before feeding the observations to the network.

# xhat = (x - μ) . (σ^2 + ε)^(-1/2)
# ^, is just z score norm with online statistics

import numpy as np

'''

Draft before I do things:

m_a = self.var * self.count  # sum of squared deviations (old)
m_b = batch_var * batch_count  # sum of squared deviations (new)
M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
self.var = M2 / total_count

'''

class RunningMeanStd:

    '''
        Goal: Track running mean and variance using Welford's parallel online algorithm
    '''

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4 # Div by 0 bad, so initially setting it to this

    # Update the stats with a batch of samples
    def update(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1) # As its a single obs, itd be a batch of 1

        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    # Chan's parallel algorithm for combining statistics.
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count # New mea being the weighted combination

        # New var being the parallel algorithm
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count # Sum of squared deviations from mean!
        self.var = M2 / total_count

        self.count = total_count

    def normalise(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8) # z-score norm

