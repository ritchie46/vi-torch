from pytest import fixture
from torch import nn
from layers.utils import accumulate_kl_div
import layers


def test_accumulate_kl_div():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                layers.LinearVariational(1, 1, 1),
                layers.LinearVariational(1, 1, 1)
            )

    model = Model()
    model.layers[0]._kl_divergence_ = 2
    model.layers[1]._kl_divergence_ = 2
    kl = accumulate_kl_div(model)
    assert kl == 4

