import torch
from torch import nn
from torch.nn.functional import pad

from .ode import stepRK4
from .nets import Phi


class OTFlow(nn.Module):

    def __init__(self, input_dim, hidden_dim, context_dim, num_layers, num_timesteps=8):
        super().__init__()
        self.phi = Phi(input_dim, hidden_dim, context_dim, num_layers)
        self.nt = num_timesteps
        self.dz = input_dim
        self.regularization_states = None

    def forward(self, z, logpx=None, context=None, reverse=False):

        if reverse:
            tspan = [1.0, 0.0]
        else:
            tspan = [0.0, 1.0]

        y, delta_log_py, aux1, aux2 = self.integrate(z, context, tspan)

        self.regularization_states = [None, aux1, aux2]
        return y, delta_log_py

    def integrate(self, x, context, tspan):
        h = (tspan[1] - tspan[0]) / self.nt

        # initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs
        z = pad(x, (0, 3, 0, 0), value=0)
        tk = tspan[0]

        for k in range(self.nt):
            z = stepRK4(z, context, self.phi, tk, tk + h)
            tk += h

        delta_log_pz = z[:, self.dz]
        aux_costs = delta_log_pz[-2:]
        z_hat = z[:, :self.dz]

        return z_hat, delta_log_pz, aux_costs[0], aux_costs[1]
