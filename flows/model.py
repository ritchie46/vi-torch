import torch
from torch import nn


class BasicFlow(nn.Module):
    def __init__(self, dim, n_flows, flow_layer):
        super().__init__()
        self.flow = nn.Sequential(*[
            flow_layer(dim) for _ in range(n_flows)
        ])
        self.mu = nn.Parameter(torch.randn(dim, ).normal_(0, 0.01))
        self.log_var = nn.Parameter(torch.randn(dim, ).normal_(1, 0.01))

    def forward(self, shape):
        std = torch.exp(0.5 * self.log_var)
        eps = torch.randn(shape)  # unit gaussian
        z0 = self.mu + eps * std

        zk = self.flow(z0)
        return z0, zk, self.mu, self.log_var