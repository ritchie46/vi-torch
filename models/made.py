from torch import nn
from layers.autoregressive import SequentialMasked, LinearMasked


class MADE(nn.Module):
    """
    See Also:
        Germain et al. (2015, Feb 12) MADE:
        Masked Autoencoder for Distribution Estimation.
        Retrieved from https://arxiv.org/abs/1502.03509
    """

    # Don't use ReLU, so that neurons don't get nullified.
    # This makes sure that the autoregressive test can verified
    def __init__(self, in_features, hidden_features):

        super().__init__()
        self.layers = SequentialMasked(
            LinearMasked(in_features, hidden_features, in_features),
            nn.ELU(),
            LinearMasked(hidden_features, hidden_features, in_features),
            nn.ELU(),
            LinearMasked(hidden_features, in_features, in_features),
            nn.Sigmoid(),
        )
        self.layers.set_mask_last_layer()

    def forward(self, x):
        return self.layers(x)



