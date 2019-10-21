from torch import nn
from layers.utils import accumulate_kl_div
import layers
import torch


def test_accumulate_kl_div():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                layers.LinearVariational(1, 1, 1), layers.LinearVariational(1, 1, 1)
            )

    model = Model()
    model.layers[0]._kl_divergence_ = 2
    model.layers[1]._kl_divergence_ = 2
    kl = accumulate_kl_div(model)
    assert kl == 4


def test_masked_linear_mask():
    """
    See Also: https://www.youtube.com/watch?v=lNW8T0W-xeE
        at 11:26
    """
    input_size = 4
    l = layers.LinearMasked(input_size, 5, input_size, bias=False)
    # Example values taken from the first hidden layer.
    l.m = torch.tensor([1, 2, 1, 2, 3])

    m_input_layer = torch.arange(1, input_size + 1)
    # last values is conditional on the previous x-values
    # and is the final prediction of the model.
    # Should not have any hidden nodes.
    m_input_layer[-1] = 1e9

    l.set_mask(m_input_layer)

    assert torch.all(
        l.mask
        == torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
            ]
        )
    )
