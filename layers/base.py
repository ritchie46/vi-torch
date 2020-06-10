from torch import nn


class LayerKL(nn.Module):
    """
    Used to aggregate KL loss.
    """
    def __init__(self):
        super().__init__()
        self._kl_divergence_: float = 0.
