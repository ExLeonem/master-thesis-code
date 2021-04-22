import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Gamma


class VariationalParams:

    def __init__(self, size, mean=None, log_std=None):

        if not mean:
            mean = np.random.randn(size)

        if not log_std:
            log_std = np.random.randn(size)

        # Variational parameters
        vp = torch.stack([mean, log_std])
        vp.requires_grad = True

        # Dimension of variational parameters
        self.size = size


    
    def dist(self):
        return Normal(self.vp[0], self.vp[1].exp())


    def rsample(self, n=torch.Size([])):
        return self.dist().rsample(n)

    
    def log_q(self, real):
        return self.dist().log_prob(real).sum()