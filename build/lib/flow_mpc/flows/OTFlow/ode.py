import torch
from torch.nn.functional import pad


def odefun(x, context, t, net):
    """
    neural ODE combining the characteristics and log-determinant (see Eq. (2)), the transport costs (see Eq. (5)), and
    the HJB regularizer (see Eq. (7)).

    d_t  [x ; l ; v ; r] = odefun( [x ; l ; v ; r] , t )

    x - particle position
    l - log determinant
    v - accumulated transport costs (Lagrangian)
    r - accumulates violation of HJB condition along trajectory
    """
    nex, d_extra = x.shape
    d = d_extra - 3

    z = pad(x[:, :d], (0, 1, 0, 0), value=t)  # concatenate with the time t
    gradPhi, trH = net.trHess(z, context)

    dx = -gradPhi[:, 0:d]
    dl = -trH.unsqueeze(1)
    dv = 0.5 * torch.sum(torch.pow(dx, 2), 1, keepdims=True)
    dr = torch.abs(-gradPhi[:, -1].unsqueeze(1) + dv)

    return torch.cat((dx, dl, dv, dr), 1)


def stepRK4(z, context, Phi, t0, t1):
    """
        Runge-Kutta 4 integration scheme
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """

    h = t1 - t0  # step size
    z0 = z
    K = h * odefun(z0, context, t0, Phi)
    z = z0 + (1.0 / 6.0) * K

    K = h * odefun(z0 + 0.5 * K, context, t0 + (h / 2), Phi)
    z += (2.0 / 6.0) * K

    K = h * odefun(z0 + 0.5 * K, context, t0 + (h / 2), Phi)
    z += (2.0 / 6.0) * K

    K = h * odefun(z0 + K, context, t0 + h, Phi)
    z += (1.0 / 6.0) * K

    return z
