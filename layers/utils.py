from torch import nn


def get_kl_layers(model):
    return [
        module
        for module in model.modules()
        if type(module) != nn.Sequential and hasattr(module, "_kl_divergence_")
    ]


def accumulate_kl_div(model):
    return sum(map(lambda module: module._kl_divergence_, get_kl_layers(model)))


def reset_kl_div(model):
    for l in get_kl_layers(model):
        l._kl_divergence_ = 0
