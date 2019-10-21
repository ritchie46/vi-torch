import torch
from torch import nn
import torch.nn.functional as F
from layers.base import KL_Layer


class IAF(KL_Layer):
    """
    Inverse Autoregressive Flow
    https://arxiv.org/pdf/1606.04934.pdf
    """
    def __init__(self, size=1, init_mu=None, init_sigma=0.01):
        super().__init__()
        if init_mu is None:
            init_mu = .5 / size
        self.s_t = nn.Parameter(torch.randn(1, size).normal_(init_mu, init_sigma))
        self.m_t = nn.Parameter(torch.randn(1, size).normal_(init_mu, init_sigma))

    def determine_log_det_jac(self, sigma_t):
        return torch.log(sigma_t + 1e-6)

    def forward(self, z):
        sigma_t = F.sigmoid(self.s_t)

        # log |det Jac|
        self._kl_divergence_ += self.determine_log_det_jac(sigma_t)

        # transformation
        return sigma_t * z + (1 - sigma_t) * self.m_t
