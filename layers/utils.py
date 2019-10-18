from torch import nn


def accumulate_kl_div(model):
    return sum(
        [
            module._kl_divergence_
            for module in model.modules()
            if type(module) != nn.Sequential and hasattr(module, "_kl_divergence_")
        ]
    )
