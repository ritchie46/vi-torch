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

    See image:
        img/made/made_tests.png
    """
    from layers.autoregressive import set_mask_output_layer
    input_size = 4
    hidden_size = 5
    input_layer = layers.LinearMasked(input_size, hidden_size, input_size, bias=False)
    # Example values taken from the first hidden layer.
    input_layer.m = torch.tensor([1, 2, 1, 2, 3])

    m_input_layer = torch.arange(1, input_size + 1)
    # last values is conditional on the previous x-values
    # and is the final prediction of the model.
    # Should not have any hidden nodes.
    m_input_layer[-1] = 1e9

    input_layer.set_mask(m_input_layer)

    assert torch.all(
        input_layer.mask
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

    # Test the masks of predefined m values.
    hidden_layer = layers.LinearMasked(hidden_size, hidden_size, input_size)
    hidden_layer.m = torch.tensor([1, 1, 2, 1, 3])
    hidden_layer.set_mask(input_layer.m)

    assert torch.all(
        hidden_layer.mask
        == torch.tensor(
            [
                [True, False, True, False, False],
                [True, False, True, False, False],
                [True, True, True, True, False],
                [True, False, True, False, False],
                [True, True, True, True, True],
            ]
        )
    )

    output_layer = layers.LinearMasked(hidden_size, input_size, input_size)
    output_layer = set_mask_output_layer(output_layer, hidden_layer.m)

    assert torch.all(
        output_layer.mask
        == torch.tensor(
            [
                [False, False, False, False, False],
                [True, True, False, True, False],
                [True, True, True, True, False],
                [True, True, True, True, True],
            ]
        )
    )


def test_sequential_masked():
    from layers.autoregressive import SequentialMasked

    torch.manual_seed(3)
    num_in = 3
    a = SequentialMasked(
        layers.LinearMasked(num_in, 5, num_in),
        nn.ReLU(),
        layers.LinearMasked(5, 3, num_in)
    )
    # Test if the mask is set on all LinearMasked layers.
    # At initializing they contain only 1's.
    assert torch.any(a[0].mask == 0)
    assert torch.any(a[-1].mask == 0)


def test_autoreggressive_made():
    # Idea from karpathy; https://github.com/karpathy/pytorch-made/blob/master/made.py
    # We predict x, and look at the partial derivatives.
    # For the autoregressive property to hold, dy/dx
    # can only be dependent of x<d. Where d is the current index.
    from models import MADE

    input_size = 10
    x = torch.ones((1, input_size))
    x.requires_grad = True

    m = MADE(in_features=input_size, hidden_features=20)

    for d in range(input_size):
        x_hat = m(x)

        # loss w.r.t. P(x_d | x_<d)
        loss = x_hat[0, d]
        loss.backward()

        assert torch.all(x.grad[0, :d] != 0)
        assert torch.all(x.grad[0, d:] == 0)

