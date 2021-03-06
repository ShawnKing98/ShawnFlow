import torch
import torch.nn as nn

__all__ = ['CouplingLayer','ResNetCouplingLayer','MaskedCouplingLayer']


class CouplingLayer(nn.Module):
    """Used in 2D experiments."""

    def __init__(self, d, intermediate_dim=128, context_dim=512, swap=False, nonlinearity='ReLu'):
        nn.Module.__init__(self)
        self.d = d - (d // 2)
        self.swap = swap
        if nonlinearity=='ReLu':
            self.nonlinearity = nn.ReLU(inplace=True)
        elif nonlinearity=='Tanh':
            self.nonlinearity = nn.Tanh()
        self.scaling_factor = nn.Parameter(torch.zeros(self.d))

        self.net_s_t = nn.Sequential(
            nn.Linear(self.d + context_dim, intermediate_dim),
            self.nonlinearity,
            nn.Linear(intermediate_dim, intermediate_dim),
            self.nonlinearity,
            nn.Linear(intermediate_dim, (d - self.d) * 2),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, logpx=None, reverse=False):
        """

        :param x: input to the layer, could be latent variable, could be original variable, depending on param reverse
        :param context: the context variable used to condition the output distribution
        :param logpx: log probability before this layer
        :param reverse: True means from original space to latent space, False means from latent space to original space
        :return y: original variable OR latent variable
        :return logpy: log probability after this layer
        """
        if self.swap:
            x = torch.cat([x[:, self.d:], x[:, :self.d]], 1)

        in_dim = self.d
        out_dim = x.shape[1] - self.d

        network_input = torch.cat((x[:, :in_dim], context), dim=1)
        s, shift = self.net_s_t(network_input).chunk(2, dim=1)
        s_fac = self.scaling_factor.exp().view(1, -1)
        scale = torch.tanh(s / s_fac) * s_fac

        ldj = torch.sum(scale.view(scale.shape[0], -1), dim=1)

        if not reverse:         # sampling
            y1 = (x[:, self.d:] - shift) * torch.exp(-scale)
            ldj = -ldj
        else:                   # training
            y1 = x[:, self.d:] * torch.exp(scale) + shift

        y = torch.cat([x[:, :self.d], y1], 1) if not self.swap else torch.cat([y1, x[:, :self.d]], 1)
        if logpx is None:
            return y
        else:
            return y, logpx + ldj

import time

class ResNetBlock(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.b1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.b2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.b1(x)
        x = x + self.act(self.fc2(x))
        x = self.b2(x)
        x = self.fc3(x)
        return x

class ResNetCouplingLayer(nn.Module):
    """Used in 2D experiments."""

    def __init__(self, d, intermediate_dim=128, context_dim=0, swap=False, nonlinearity='ReLu'):
        nn.Module.__init__(self)
        self.d = d - (d // 2)
        self.swap = swap
        if nonlinearity == 'ReLu':
            self.nonlinearity = nn.ReLU(inplace=True)
        elif nonlinearity == 'Tanh':
            self.nonlinearity = nn.Tanh()
        self.scaling_factor = nn.Parameter(torch.zeros(d//2))

        self.net_s_t = nn.Sequential(
            nn.Linear(self.d + context_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            self.nonlinearity,
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            self.nonlinearity,
            nn.Linear(intermediate_dim, (d - self.d) * 2),
        )
        '''self.net_s_t = ResNetBlock(self.d + context_dim, self.d*2, intermediate_dim)'''

        nonlinearity_name = 'relu' if nonlinearity == 'ReLu' else 'tanh'
        nn.init.kaiming_normal_(self.net_s_t[0].weight, nonlinearity=nonlinearity_name)
        nn.init.kaiming_normal_(self.net_s_t[3].weight, nonlinearity=nonlinearity_name)
        nn.init.kaiming_normal_(self.net_s_t[6].weight, nonlinearity='linear')
        # nn.init.xavier_uniform_(self.net_s_t[0].weight,
        #                         gain=nn.init.calculate_gain(nonlinearity_name)/10)
        # nn.init.xavier_uniform_(self.net_s_t[3].weight,
        #                         gain=nn.init.calculate_gain(nonlinearity_name)/10)
        # nn.init.xavier_uniform_(self.net_s_t[6].weight,
        #                         gain=nn.init.calculate_gain('linear')/10)

    def forward(self, x, context=None, logpx=None, reverse=False):
        start = time.time()
        if self.swap:
            x = torch.cat([x[:, self.d:], x[:, :self.d]], 1)

        in_dim = self.d
        out_dim = x.shape[1] - self.d
        if context is not None:
            network_input = torch.cat((x[:, :in_dim], context), dim=1)
        else:
            network_input = x[:, :in_dim]

        s_t = self.net_s_t(network_input)
        #scale = torch.tanh(s_t[:, :out_dim]) + 1.0

        # Stabilize scaling output
        s_fac = self.scaling_factor.exp().view(1, -1)
        scale = torch.tanh(s_t[:, :out_dim] / s_fac) * s_fac

        shift = s_t[:, out_dim:]

        #logdetjac = torch.sum(torch.log(scale).view(scale.shape[0], -1), dim=1)
        ldj = torch.sum(scale, dim=1)
        #if not reverse:
        #    y1 = x[:, self.d:] * scale + shift
        #else:
        #    y1 = (x[:, self.d:] - shift) / scale

        # Affine transformation
        if not reverse:
            y1 = (x[:, self.d:] - shift) * torch.exp(-1 * scale)
            ldj = - ldj
        else:
            y1 = x[:, self.d:] * torch.exp(scale) + shift

        y = torch.cat([x[:, :self.d], y1], 1) if not self.swap else torch.cat([y1, x[:, :self.d]], 1)

        end = time.time()

        if logpx is None:
            return y
        else:
            return y, logpx + ldj

class MaxLayer(nn.Module):

    """ Saturates layer to maximum and minimum value, assumed symmetric about zero
     -- ensures samples from flow are within a certain range
    -- essentially makes flow have bounded support"""

    def __init__(self, max_value):
        super().__init__()
        self.max_value = max_value

    def forward(self, x, context=None, logpx=None, reverse=False):

        if reverse:
            x_normalised = (x + 0.5) / (2 * self.max_value)

            z = torch.logit(x_normalised, eps=1e-6)
            ldj = torch.log(
                2 * self.max_value * torch.sum(x_normalised * (1 - x_normalised), dim=1, keepdim=True))

        else:
            z = 2 * self.max_value * (torch.sigmoid(x) - 0.5)
            ldj = torch.log(2 * self.max_value * torch.sum(torch.sigmoid(x) * (1 - torch.sigmoid(x)), dim=1, keepdim=True))

        if logpx is not None:
            logpx = logpx + ldj
        return z, logpx


class RationalQuadraticSplineCouplingLayer(nn.Module):
    """
    Neural spline flow coupling layer, wrapper for the implementation
    of Durkan et al., see https://github.com/bayesiains/nsf
    """
    def __init__(
            self,
            d,
            intermediate_dim=128,
            context_dim=0,
            swap=False,
            nonlinearity='ReLu',
            num_bins=8,
            tail_bound=3
    ):
        """
        Constructor
        :param num_input_channels: Flow dimension
        :type num_input_channels: Int
        :param num_blocks: Number of residual blocks of the parameter NN
        :type num_blocks: Int
        :param num_hidden_channels: Number of hidden units of the NN
        :type num_hidden_channels: Int
        :param num_bins: Number of bins
        :type num_bins: Int
        :param tail_bound: Bound of the spline tails
        :type tail_bound: Int
        :param activation: Activation function
        :type activation: torch module
        :param dropout_probability: Dropout probability of the NN
        :type dropout_probability: Float
        :param reverse_mask: Flag whether the reverse mask should be used
        :type reverse_mask: Boolean
        :param reverse: Flag whether forward and backward pass shall be swapped
        :type reverse: Boolean
        """
        super().__init__()
        self.net_s_t = nn.Sequential(
            nn.Linear(self.d + context_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            self.nonlinearity,
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            self.nonlinearity,
            nn.Linear(intermediate_dim, (d - self.d) * 2),
        )
        '''self.net_s_t = ResNetBlock(self.d + context_dim, self.d*2, intermediate_dim)'''

        nonlinearity_name = 'relu' if nonlinearity == 'ReLu' else 'tanh'
        nn.init.kaiming_normal_(self.net_s_t[0].weight, nonlinearity=nonlinearity_name)
        nn.init.kaiming_normal_(self.net_s_t[3].weight, nonlinearity=nonlinearity_name)
        nn.init.kaiming_normal_(self.net_s_t[6].weight, nonlinearity='linear')

    def forward(self, z):
        if self.reverse:
            z, log_det = self.prqct.inverse(z)
        else:
            z, log_det = self.prqct(z)
        return z, log_det.view(-1)

    def inverse(self, z):
        if self.reverse:
            z, log_det = self.prqct(z)
        else:
            z, log_det = self.prqct.inverse(z)
        return z, log_det.view(-1)
