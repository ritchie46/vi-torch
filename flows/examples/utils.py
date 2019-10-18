import torch
import torch.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
from flows.model import BasicFlow


def target_density(z):
    z1, z2 = z[..., 0], z[..., 1]
    norm = (z1 ** 2 + z2 ** 2) ** 0.5
    exp1 = torch.exp(-0.2 * ((z1 - 2) / 0.8) ** 2)
    exp2 = torch.exp(-0.2 * ((z1 + 2) / 0.8) ** 2)
    u = 0.5 * ((norm - 4) / 0.4) ** 2 - torch.log(exp1 + exp2)
    return torch.exp(-u)


def det_loss(mu, log_var, z_0, z_k, ldj, beta):
    # Note that I assume uniform prior here.
    # So P(z) is constant and not modelled in this loss function
    batch_size = z_0.size(0)

    # Qz0
    log_qz0 = dist.Normal(mu, torch.exp(0.5 * log_var)).log_prob(z_0)
    # Qzk = Qz0 + sum(log det jac)
    log_qzk = log_qz0.sum() - ldj.sum()
    # P(x|z)
    nll = -torch.log(target_density(z_k) + 1e-7).sum() * beta
    return (log_qzk + nll) / batch_size


def train_flow(flow, sample_shape, epochs=1000):
    optim = torch.optim.Adam(flow.parameters(), lr=1e-2)

    for i in range(epochs):
        z0, zk, ldj, mu, log_var = flow(shape=sample_shape)
        loss = det_loss(mu=mu, log_var=log_var, z_0=z0, z_k=zk, ldj=ldj, beta=1)
        loss.backward()
        optim.step()
        optim.zero_grad()
        if i % 100 == 0:
            print(loss.item())


def run_example(flow_layer, n_flows=8):
    x1 = np.linspace(-7.5, 7.5)
    x2 = np.linspace(-7.5, 7.5)
    x1_s, x2_s = np.meshgrid(x1, x2)
    x_field = np.concatenate([x1_s[..., None], x2_s[..., None]], axis=-1)
    x_field = torch.tensor(x_field, dtype=torch.float)

    plt.figure(figsize=(8, 8))
    plt.title("Target distribution")
    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    plt.contourf(x1_s, x2_s, target_density(x_field))
    plt.show()

    def show_samples(s):
        plt.figure(figsize=(6, 6))
        plt.scatter(s[:, 0], s[:, 1], alpha=0.1)
        plt.show()

    flow = BasicFlow(dim=2, n_flows=n_flows, flow_layer=flow_layer)

    # batch, dim
    sample_shape = (50, 2)
    train_flow(flow, sample_shape, epochs=2500)
    z0, zk, ldj, mu, log_var = flow((5000, 2))
    show_samples(zk.data)
