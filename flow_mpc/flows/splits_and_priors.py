import torch
from torch import nn
import numpy as np
from torch.nn import functional as F, init
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal


class SqueezeFlow(nn.Module):

    def forward(self, z, logpx, reverse=False):
        B, C, H, W = z.shape
        if not reverse:
            # Forward direction: H x W x C => H/2 x W/2 x 4C
            z = z.reshape(B, C, H // 2, 2, W // 2, 2)
            z = z.permute(0, 1, 3, 5, 2, 4)
            z = z.reshape(B, 4 * C, H // 2, W // 2)
        else:
            # Reverse direction: H/2 x W/2 x 4C => H x W x C
            z = z.reshape(B, C // 4, 2, 2, H, W)
            z = z.permute(0, 1, 4, 2, 5, 3)
            z = z.reshape(B, C // 4, H * 2, W * 2)
        return z, logpx


class SqueezeVoxelFlow(nn.Module):

    def forward(self, z, logpx, reverse=False):
        B, C, D, H, W = z.shape
        if not reverse:
            # Forward direction: H x W x C => H/2 x W/2 x 4C
            z = z.reshape(B, C, D // 2, 2, H // 2, 2, W // 2, 2)
            z = z.permute(0, 1, 3, 5, 7, 2, 4, 6)
            z = z.reshape(B, 8 * C, D // 2, H // 2, W // 2)
        else:
            # Reverse direction: H/2 x W/2 x 4C => H x W x C
            z = z.reshape(B, C // 8, 2, 2, 2, D, H, W)
            z = z.permute(0, 1, 5, 2, 6, 3, 7, 4)
            z = z.reshape(B, C // 8, D * 2, H * 2, W * 2)
        return z, logpx

class GaussianPrior(nn.Module):
    """
    A Gaussian prior with mean and covariance being parameterized, the parameterization of covariance matrix is by Cholesky decomposition
    """
    def __init__(self, z_dim, eps=1e-3, identity_init=True):
        super(GaussianPrior, self).__init__()
        self.z_dim = z_dim
        self.eps = eps

        self.lower_indices = np.tril_indices(z_dim, k=-1)
        self.diag_indices = np.diag_indices(z_dim)

        n_triangular_entries = ((z_dim - 1) * z_dim) // 2

        self.mu = nn.Parameter(torch.zeros(z_dim))
        self.lower_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.unconstrained_diag = nn.Parameter(torch.zeros(z_dim))

        self._initialize(identity_init)

    def _initialize(self, identity_init):
        init.zeros_(self.mu)
        if identity_init:
            init.zeros_(self.lower_entries)
            constant = np.log(np.exp(1 - self.eps) - 1)
            init.constant_(self.unconstrained_diag, constant)
        else:
            stdv = 1.0 / np.sqrt(self.features)
            init.uniform_(self.lower_entries, -stdv, stdv)
            init.uniform_(self.unconstrained_diag, -stdv, stdv)

    @property
    def diag(self):
        return F.softplus(self.unconstrained_diag) + self.eps

    @property
    def std(self):
        lower = self.lower_entries.new_zeros(self.z_dim, self.z_dim)
        lower[self.lower_indices[0], self.lower_indices[1]] = self.lower_entries
        lower[self.diag_indices[0], self.diag_indices[1]] = self.diag

        return lower @ lower.t()

    def forward(self, z, logpx, context, reverse=False):
        batch_size = context.shape[0]
        # z_mu, z_std = self.mu, self.std
        z_mu, z_std = torch.zeros(160, device='cuda', dtype=torch.double), torch.eye(160, device='cuda', dtype=torch.double)
        prior = MultivariateNormal(z_mu, z_std)
        if not reverse:
            if z is None:
                z = prior.rsample((batch_size,))

        logpx = logpx + prior.log_prob(z)
        return z, logpx

class ConditionalPrior(nn.Module):

    def __init__(self, context_dim, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(context_dim, context_dim)
        self.fc2 = nn.Linear(context_dim, 2 * z_dim)
        self.act_fn = F.relu

    def forward(self, z, logpx, context, reverse=False):
        z_mu, z_std = torch.chunk(self.fc2(self.act_fn(self.fc1(context))), chunks=2, dim=1)
        z_std = torch.sigmoid(z_std) + 1e-7

        prior = Normal(z_mu, z_std)
        if not reverse:
            if z is None:
                z = prior.rsample()

        logpx = logpx + prior.log_prob(z).sum(dim=1)
        return z, logpx


class ConditionalSplitFlow(nn.Module):

    def __init__(self, z_dim, z_split_dim, context_dim):
        super().__init__()
        self.fc1 = nn.Linear(context_dim + z_dim - z_split_dim, context_dim)
        self.fc2 = nn.Linear(context_dim, 2 * z_split_dim)
        self.z_split_dim = z_split_dim
        self.act_fn = F.relu

    def forward(self, z, logpx, context, reverse=False):
        inpt = torch.cat((z[:, :self.z_split_dim], context), dim=1)
        z_split_mu, z_split_std = torch.chunk(self.fc2(self.act_fn(self.fc1(inpt))), chunks=2, dim=1)
        z_split_std = torch.sigmoid(z_split_std) + 1e-2
        prior = Normal(z_split_mu, z_split_std)

        if not reverse:
            z_split = prior.rsample()
            z = torch.cat((z, z_split), dim=1)
        else:
            z_split = z[:, -self.z_split_dim:]
            z = z[:, :self.z_split_dim]

        logpx = logpx + prior.log_prob(z_split).sum(dim=1)
        return z, logpx


class SplitFlow(nn.Module):

    def __init__(self):
        super().__init__()
        self.prior = Normal(loc=0.0, scale=1.0)

    def forward(self, z, logpx, reverse=False):
        B = z.shape[0]
        if not reverse:
            z, z_split = z.chunk(2, dim=1)
        else:
            z_split = self.prior.sample(sample_shape=z.shape).to(z.device)
            z = torch.cat([z, z_split], dim=1)
        logpx += self.prior.log_prob(z_split).reshape(B, -1).sum(dim=1)
        return z, logpx
