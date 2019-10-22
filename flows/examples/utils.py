import torch
import torch.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
from flows.model import BasicFlow
from layers.utils import accumulate_kl_div, reset_kl_div
from flows import targets


def target_density(z, name="u2"):
    f = getattr(targets, name)
    return f(z)


def det_loss(mu, log_var, z_0, z_k, ldj, beta, target_function_name):
    # Note that I assume uniform prior here.
    # So P(z) is constant and not modelled in this loss function
    batch_size = z_0.size(0)

    # Qz0
    log_qz0 = dist.Normal(mu, torch.exp(0.5 * log_var)).log_prob(z_0)
    # Qzk = Qz0 + sum(log det jac)
    log_qzk = log_qz0.sum() - ldj.sum()
    # P(x|z)
    nll = -torch.log(target_density(z_k, target_function_name) + 1e-7).sum() * beta
    return (log_qzk + nll) / batch_size


def train_flow(flow, sample_shape, epochs=1000, target_function_name="ta"):
    optim = torch.optim.Adam(flow.parameters(), lr=1e-2)

    for i in range(epochs):
        z0, zk, mu, log_var = flow(shape=sample_shape)
        ldj = accumulate_kl_div(flow)

        loss = det_loss(
            mu=mu,
            log_var=log_var,
            z_0=z0,
            z_k=zk,
            ldj=ldj,
            beta=1,
            target_function_name=target_function_name,
        )
        loss.backward()
        optim.step()
        optim.zero_grad()
        reset_kl_div(flow)
        if i % 100 == 0:
            print(loss.item())


def run_example(
    flow_layer, n_flows=8, epochs=2500, samples=50, target_function_name="ta"
):
    x1 = np.linspace(-7.5, 7.5)
    x2 = np.linspace(-7.5, 7.5)
    x1_s, x2_s = np.meshgrid(x1, x2)
    x_field = np.concatenate([x1_s[..., None], x2_s[..., None]], axis=-1)
    x_field = torch.tensor(x_field, dtype=torch.float)
    x_field = x_field.reshape(x_field.shape[0] * x_field.shape[1], 2)

    plt.figure(figsize=(8, 8))
    plt.title("Target distribution")
    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    plt.contourf(
        x1_s,
        x2_s,
        target_density(x_field, target_function_name).reshape(x1.shape[0], x1.shape[0]),
    )
    plt.show()

    def show_samples(s):
        plt.figure(figsize=(6, 6))
        plt.scatter(s[:, 0], s[:, 1], alpha=0.1)
        plt.show()

    flow = BasicFlow(dim=2, n_flows=n_flows, flow_layer=flow_layer)

    # batch, dim
    sample_shape = (samples, 2)
    train_flow(
        flow, sample_shape, epochs=epochs, target_function_name=target_function_name
    )
    z0, zk, mu, log_var = flow((5000, 2))
    show_samples(zk.data)
