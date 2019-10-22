import torch
import math

# synthetic data in table 1 of paper:
# Rezende & Mohamed. (2015, May 21)
# Variational Inference with Normalizing Flows.
# Retrieved from https://arxiv.org/abs/1505.05770

# Code provided by:
# Weixsong
# Variational Inference with Normalizing Flows
# https://github.com/weixsong/NormalizingFlow/blob/master/synthetic_data.py

# All the functions needed to be transform by exp(-x) to have the right probability density.


def ta(z):
    z1, z2 = z[..., 0], z[..., 1]
    norm = (z1 ** 2 + z2 ** 2) ** 0.5
    exp1 = torch.exp(-0.2 * ((z1 - 2) / 0.8) ** 2)
    exp2 = torch.exp(-0.2 * ((z1 + 2) / 0.8) ** 2)
    u = 0.5 * ((norm - 4) / 0.4) ** 2 - torch.log(exp1 + exp2)
    return torch.exp(-u)


def w1(z):
    return torch.sin(2 * math.pi * z[:, 0] / 4)


def w2(z):
    return 3 * torch.exp(-0.5 * ((z[..., 0] - 1) / 0.6) ** 2)


def w3(z):
    return 3 * (1.0 / (1 + torch.exp(-(z[..., 0] - 1) / 0.3)))


def u1(z):
    add1 = 0.5 * ((torch.norm(z, 2, 1) - 2) / 0.4) ** 2
    add2 = -torch.log(torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2) + torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2))
    return torch.exp(-(add1 + add2))


def u2(z):
    z1, z2 = z[..., 0], z[..., 1]
    return torch.exp(-0.5 * ((z2 - w1(z)) / 0.4) ** 2)


def u3(z):
    in1 = torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2)
    in2 = torch.exp(-0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2)
    return torch.exp(torch.log(in1 + in2 + 1e-9))


def u4(z):
    in1 = torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2)
    in2 = torch.exp(-0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2)
    return torch.exp(torch.log(in1 + in2))