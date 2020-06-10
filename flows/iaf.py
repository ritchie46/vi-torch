import torch
from torch import nn
import torch.nn.functional as F
from layers.base import LayerKL
from models import MADE


class AutoRegressiveNN(MADE):
    def __init__(self, in_features, hidden_features, context_features):
        super().__init__(in_features, hidden_features)
        self.context = nn.Linear(context_features, in_features)
        # remove MADE output layer
        del self.layers[len(self.layers) - 1]

    def forward(self, z, h):
        return self.layers(z) + self.context(h)


class IAF(LayerKL):
    """
    Inverse Autoregressive Flow
    https://arxiv.org/pdf/1606.04934.pdf
    """

    def __init__(self, size=1, context_size=1, auto_regressive_hidden=1):
        super().__init__()
        self.context_size = context_size
        self.s_t = AutoRegressiveNN(
            in_features=size,
            hidden_features=auto_regressive_hidden,
            context_features=context_size,
        )
        self.m_t = AutoRegressiveNN(
            in_features=size,
            hidden_features=auto_regressive_hidden,
            context_features=context_size,
        )

    def determine_log_det_jac(self, sigma_t):
        return torch.log(sigma_t + 1e-6).sum(1)

    def forward(self, z, h=None):
        if h is None:
            h = torch.zeros(self.context_size)

        # Initially s_t should be large, i.e. 1 or 2.
        s_t = self.s_t(z, h) + 1.5
        sigma_t = F.sigmoid(s_t)
        m_t = self.m_t(z, h)

        # log |det Jac|
        self._kl_divergence_ += self.determine_log_det_jac(sigma_t)

        # transformation
        return sigma_t * z + (1 - sigma_t) * m_t
