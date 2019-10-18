import torch
from torch import nn


class KL_Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self._kl_divergence_ = 0