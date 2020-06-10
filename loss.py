from layers.utils import accumulate_kl_div
import torch
from custom_types import FloatTensor


def variational_free_energy(model, criterion, y_pred, target, beta=1.):
    """

    Parameters
    ----------
    model : nn.Module
        With layers that subclass. layers.base.KL_Layer
    criterion : function
        Loss aggregation should be sum.
    y_pred : tensor
    target : tensor
    beta : float
        Annealing factor.
        Performances can be better if annealling factor slowly increments to 1.

    Returns
    -------

    """

    reconstruction_error = criterion(y_pred, target)
    kl = accumulate_kl_div(model)
    return reconstruction_error + beta * kl


def analytical_kl(mu: FloatTensor, log_var: FloatTensor) -> FloatTensor:
    """
    Analytical solution when prior
        P(z) = N(0, 1)
        and variational distribution.
        Q_theta(z|x) is Gaussian

    Parameters
    ----------
    mu
        variational parameter mu of N(mu, sigma)
    log_var
        variational parameter log(variance) of N(mu, sigma)

    Returns
    -------
    KL divergence

    """
    return (-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp())).sum()