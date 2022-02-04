import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal


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


class ConditionalPrior(nn.Module):

    def __init__(self, context_dim, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(context_dim, context_dim)
        self.fc2 = nn.Linear(context_dim, 2 * z_dim)
        self.act_fn = F.relu

    def forward(self, z, logpx, context, reverse=False):
        z_mu, z_std = torch.chunk(self.fc2(self.act_fn(self.fc1(context))), chunks=2, dim=1)
        z_std = torch.sigmoid(z_std) + 1e-2

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
