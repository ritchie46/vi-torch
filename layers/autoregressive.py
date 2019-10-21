from torch import nn
import torch
import torch.nn.functional as F


class MaskedLinear(nn.Module):
    """
    Masked Linear layers used in Made.

    See Also:
        Germain et al. (2015, Feb 12) MADE:
        Masked Autoencoder for Distribution Estimation.
        Retrieved from https://arxiv.org/abs/1502.03509

    """
    def __init__(self, in_features, out_features, num_features, bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        # m function of the paper. Every hidden node, gets a number between 1 and D-1
        self.m = torch.randint(1, num_features, size=(out_features,)).type(torch.int32)
        self.register_buffer(
            "mask", torch.ones_like(self.linear.weight).type(torch.uint8)
        )

    def set_mask(self, m_previous_layer):
        """
        Sets mask matrix of the current layer.

        Parameters
        ----------
        m_previous_layer : tensor
            m values for previous layer layer.
            The first layers should be incremental except for the last value,
            as the model does not make a prediction P(x_D+1 | x_<D + 1).
            The last prediction is P(x_D| x_<D)
        """
        self.mask[...] = (m_previous_layer[:, None] <= self.m[None, :]).T

    def forward(self, x):
        if self.linear.bias is None:
            b = 0
        else:
            b = self.linear.bias

        return F.linear(x, self.linear.weight * self.mask + b)


