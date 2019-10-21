from layers.utils import accumulate_kl_div


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