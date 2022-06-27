import ipdb
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F, init
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from .autoregressive import MaskLinear


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
    A Gaussian prior with mean and covariance *NOT* being parameterized, the parameterization of covariance matrix is by Cholesky decomposition
    """
    def __init__(self, z_dim, eps=1e-7, identity_init=True):
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
    def lower(self):
        lower = self.lower_entries.new_zeros(self.z_dim, self.z_dim)
        lower[self.lower_indices[0], self.lower_indices[1]] = self.lower_entries
        lower[self.diag_indices[0], self.diag_indices[1]] = self.diag

        return lower

    def forward(self, z, logpx, context, reverse=False):
        batch_size = context.shape[0]
        # z_mu, z_lower = self.mu, self.lower
        z_mu, z_lower = torch.zeros(self.z_dim, device='cuda', dtype=torch.float), torch.eye(self.z_dim, device='cuda', dtype=torch.float)
        prior = MultivariateNormal(loc=z_mu, scale_tril=z_lower)
        if not reverse:
            if z is None:
                z = prior.rsample((batch_size,))
                # z = z_mu

        logpx = logpx + prior.log_prob(z)
        return z, logpx

class ConditionalPrior(nn.Module):
    """
    A conditional Gaussian prior with mean and covariance generated by an MLP, the covariance matrix is represented by Cholesky decomposition
    """
    def __init__(self, context_dim, z_dim, hidden_dim=256, hidden_layer_num=2, context_order: torch.Tensor=None):
        super().__init__()
        self.eps = 1e-7
        # self.eps = 0
        self.z_dim = z_dim
        self.lower_indices = np.tril_indices(z_dim, k=-1)
        self.diag_indices = np.diag_indices(z_dim)
        n_triangular_entries = ((z_dim - 1) * z_dim) // 2
        if context_order is None:
            fc = [nn.Linear(context_dim, hidden_dim),
                  nn.ReLU()]
            for _ in range(hidden_layer_num-1):
                fc.append(nn.Linear(hidden_dim, hidden_dim))
                fc.append(nn.ReLU())
            fc.append(nn.Linear(hidden_dim, 2 * z_dim))
            self.fc = nn.Sequential(*fc)
            # for i in range(hidden_layer_num+1):
            #     self.register_buffer(f"mask_{i}", None)
            # self.fc = nn.Sequential(
            #     nn.Linear(context_dim, hidden_dim),
            #     nn.ReLU(),
            #     nn.Linear(hidden_dim, hidden_dim),
            #     nn.ReLU(),
            #     # nn.Linear(hidden_dim, 2 * z_dim + n_triangular_entries)
            #     nn.Linear(hidden_dim, 2 * z_dim)
            # )
        else:
            assert hidden_dim >= context_dim
            max_order = int(context_order.max())
            element_orders = [context_order,
                              torch.cat([torch.randperm(max_order) for _ in range(hidden_dim//max_order)] + [torch.randperm(hidden_dim%(max_order))])]
            layers = [MaskLinear(context_dim, hidden_dim)]
            for _ in range(hidden_layer_num - 1):
                layers.append(MaskLinear(hidden_dim, hidden_dim))
                element_orders.append(torch.cat([torch.randperm(max_order) for _ in range(hidden_dim//max_order)] + [torch.randperm(hidden_dim%(max_order))]))
            layers.append(MaskLinear(hidden_dim, 2 * z_dim))
            element_orders.append(torch.cat([torch.arange(z_dim) for _ in range(2)]))
            # self.fc = nn.Sequential(*fc)
            fc = []
            for i, layer in enumerate(layers):
                in_order = element_orders[i].repeat(layer.out_features, 1)
                out_order = element_orders[i + 1].unsqueeze(1).repeat(1, layer.in_features)
                mask = out_order >= in_order
                layer.mask = mask
                fc.append(layer)
                fc.append(nn.ReLU())
                # self.register_buffer(f'mask_{i}', mask)
            self.fc = nn.Sequential(*fc[0:-1])

    def mu_lower(self, context):
        # Deprecated
        batch_size = context.shape[0]
        mlp_out = self.fc(context)
        z_mu = mlp_out[:, :self.z_dim]
        unconstrained_diag = mlp_out[:, self.z_dim: 2 * self.z_dim]
        diag = F.softplus(unconstrained_diag) + self.eps
        lower_entries = mlp_out[:, 2 * self.z_dim:]

        lower = z_mu.new_zeros(batch_size, self.z_dim, self.z_dim)
        lower[:, self.lower_indices[0], self.lower_indices[1]] = lower_entries
        lower[:, self.diag_indices[0], self.diag_indices[1]] = diag
        ipdb.set_trace()
        return z_mu, lower

    def forward(self, z, logpx, context, reverse=False):
        # z_mu, lower = self.mu_lower(context)
        # prior = MultivariateNormal(loc=z_mu, scale_tril=lower)
        prior_mu, prior_std = torch.chunk(self.fc(context), chunks=2, dim=1)

        prior_std = torch.sigmoid(prior_std) + self.eps
        prior = Normal(prior_mu, prior_std)

        if not reverse:
            if z is None:
                # z = prior_mu
                z = prior.rsample()
        # try:
        logpx = logpx + prior.log_prob(z).sum(1)
        # except:
        #     import ipdb
        #     ipdb.set_trace()
        return z, logpx


class ConditionalSplitFlow(nn.Module):

    def __init__(self, z_dim, z_split_dim, context_dim, hidden_dim=256, context_mask=None):
        super().__init__()
        self.fc = [nn.Linear(context_dim + z_dim - z_split_dim, hidden_dim),
                   nn.ReLU(),
                   nn.Linear(hidden_dim, hidden_dim),
                   nn.ReLU(),
                   nn.Linear(hidden_dim, 2 * z_split_dim)]
        self.fc = nn.Sequential(*self.fc)
        # self.fc2 = nn.Linear(context_dim, 2 * z_split_dim)
        self.z_split_dim = z_split_dim
        self.register_buffer("context_mask", context_mask)
        # self.act_fn = F.relu

    def forward(self, z, logpx, context, reverse=False):
        """Keep the first z_split_dim variable and drop the remaining ones"""
        if self.context_mask is not None:
            context = context[self.context_mask.bool()]
        inpt = torch.cat((z[:, :self.z_split_dim], context), dim=1)
        # z_split_mu, z_split_std = torch.chunk(self.fc2(self.act_fn(self.fc1(inpt))), chunks=2, dim=1)
        z_split_mu, z_split_std = torch.chunk(self.fc(inpt), chunks=2, dim=1)
        z_split_std = torch.sigmoid(z_split_std) + 1e-7
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
