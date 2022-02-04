import torch
from torch import nn
from torch.nn import functional as F

from flow_mpc.flows import ResNetCouplingLayer as CouplingLayer, BatchNormFlow
from flow_mpc.flows import RandomPermutation, LULinear
from flow_mpc.flows.sequential import SequentialFlow

from flow_mpc.action_samplers.flow import build_ffjord, build_nvp_flow


class FlowPrior(nn.Module):

    def __init__(self, prior_dim, type='nvp'):
        super().__init__()
        self.dz = prior_dim
        if type == 'nvp':
            self.flow = build_nvp_flow(prior_dim, 0, 4)
        elif type == 'ffjord':
            self.flow = build_ffjord(prior_dim, 0, 1)
        self.flow_type = type
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        self.dummy_param = nn.Parameter(torch.empty(0))

    def log_prob(self, x):
        log_px = 0
        z, delta_log_pz = self.flow(x, logpx=log_px, reverse=True)
        log_pz = self.prior.log_prob(z).sum(dim=1) - delta_log_pz

        # regularization penalty
        if self.training and self.flow_type == 'ffjord':
            r1, r2 = self.flow.chain[0].regularization_states
            log_pz = log_pz - 0.01 * r1 - 0.01 * r2

        return log_pz.unsqueeze(1)


    def sample(self, sample_shape=None):
        if sample_shape is None:
            sample_shame = (1, self.dz)
        assert sample_shape[1] == self.dz
        logpz = 0
        z = self.prior.sample(sample_shape=sample_shape).to(device=self.dummy_param.device)
        z, log_pz = self.flow(z, logpx=logpz, reverse=False)
        return z

class VAE(nn.Module):
    def __init__(self, latent_dim, num_channels=[32, 64, 128, 256], flow_prior=None, voxels=False):
        flow_prior_choices = [None, 'nvp', 'ffjord']
        if flow_prior not in flow_prior_choices:
            raise ValueError('Invalid choice of flow prior')
        super().__init__()
        if voxels:
            self.encoder = Conv3DEncoder(latent_dim, num_channels)
            self.decoder = Conv3DDecoder(latent_dim, num_channels[::-1])
        else:
            self.encoder = ConvEncoder(latent_dim, num_channels)
            self.decoder = ConvDecoder(latent_dim, num_channels[::-1])

        self.use_flow_prior = flow_prior is not None
        if self.use_flow_prior:
            self.prior = FlowPrior(latent_dim, type=flow_prior)
        else:
            self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def encode(self, x):
        latent_mu, latent_sigma = self.encoder(x)
        latent = latent_mu + latent_sigma * torch.randn_like(latent_mu)
        return latent, latent_mu, latent_sigma

    def forward(self, x):
        latent, latent_mu, latent_sigma = self.encode(x)
        x_hat = self.decoder(latent)
        return x_hat, latent, latent_mu, latent_sigma

    def get_kl_divergence(self, latent, latent_mu, latent_sigma, use_samples=False):
        q = torch.distributions.normal.Normal(latent_mu, latent_sigma)

        if self.use_flow_prior or use_samples:
            log_pz = self.prior.log_prob(latent)
            return q.log_prob(latent).sum(dim=1) - log_pz.sum(dim=1)

        return torch.distributions.kl.kl_divergence(q, self.prior).sum(dim=1)


class Conv3DEncoder(nn.Module):

    def __init__(self, latent_dim, num_channels = [32, 64, 128, 256]):
        super().__init__()
        self.act_fn = F.relu
        self.conv_net = nn.Sequential(
            nn.Conv3d(1, num_channels[0], (3, 3, 3), stride=2, padding=1),
            #nn.BatchNorm3d(num_channels[0]),
            nn.ReLU(),
            nn.Conv3d(num_channels[0], num_channels[1], (3, 3, 3), stride=2, padding=1),
            #nn.BatchNorm3d(num_channels[1]),
            nn.ReLU(),
            nn.Conv3d(num_channels[1], num_channels[2], (3, 3, 3), stride=2, padding=1),
            #nn.BatchNorm3d(num_channels[2]),
            nn.ReLU(),
            nn.Conv3d(num_channels[2], num_channels[3], (3, 3, 3), stride=2, padding=1),
            #nn.BatchNorm3d(num_channels[3]),
            nn.ReLU()
        )
        self.h_dim = int(num_channels[3] * 4 * 4 * 4)
        self.fc1 = nn.Linear(self.h_dim, latent_dim * 2)

    def forward(self, x):
        h = self.conv_net(x)
        h = self.fc1(h.view(-1, self.h_dim))

        latent_mu, h_sigma = torch.chunk(h, dim=1, chunks=2)
        latent_sigma = F.softplus(h_sigma) + 1e-2
        return latent_mu, latent_sigma

class Conv3DDecoder(nn.Module):
    def __init__(self, latent_dim, num_channels = [256, 128, 64, 32]):
        super().__init__()
        self.act_fn = F.relu
        self.h_dim = int(num_channels[0] * 4 * 4 * 4)
        self.first_conv_dim = (num_channels[0], 4, 4, 4)
        self.fc1 = nn.Linear(latent_dim, self.h_dim)

        self.deconv_net = nn.Sequential(
            nn.ConvTranspose3d(num_channels[0], num_channels[1], (4, 4, 4), stride=2, padding=1),
            #nn.BatchNorm3d(num_channels[1]),
            nn.ReLU(),
            nn.ConvTranspose3d(num_channels[1], num_channels[2], (4, 4, 4), stride=2, padding=1),
            #nn.BatchNorm3d(num_channels[2]),
            nn.ReLU(),
            nn.ConvTranspose3d(num_channels[2], num_channels[3], (4, 4, 4), stride=2, padding=1),
            #nn.BatchNorm3d(num_channels[3]),
            nn.ReLU(),
            nn.ConvTranspose3d(num_channels[3], 1, (4, 4, 4), stride=2, padding=1)
        )

    def forward(self, latent):
        h = self.act_fn(self.fc1(latent)).view(-1, *self.first_conv_dim)
        return self.deconv_net(h)

class ConvEncoder(nn.Module):

    def __init__(self, latent_dim, num_channels = [32, 64, 128, 256]):
        super().__init__()
        self.act_fn = F.relu
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, num_channels[0], (3, 3), stride=2, padding=1),
            #nn.BatchNorm2d(num_channels[0]),
            nn.ReLU(),
            nn.Conv2d(num_channels[0], num_channels[1], (3, 3), stride=2, padding=1),
            #nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(),
            nn.Conv2d(num_channels[1], num_channels[2], (3, 3), stride=2, padding=1),
            #nn.BatchNorm2d(num_channels[2]),
            nn.ReLU(),
            nn.Conv2d(num_channels[2], num_channels[3], (3, 3), stride=2, padding=1),
            #nn.BatchNorm2d(num_channels[3]),
            nn.ReLU()
        )
        self.h_dim = int(num_channels[3] * 4 * 4)
        self.fc1 = nn.Linear(self.h_dim, latent_dim * 2)

    def forward(self, x):
        h = self.conv_net(x)
        h = self.fc1(h.view(-1, self.h_dim))

        latent_mu, h_sigma = torch.chunk(h, dim=1, chunks=2)
        latent_sigma = F.softplus(h_sigma) + 1e-2
        return latent_mu, latent_sigma

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, num_channels = [256, 128, 64, 32]):
        super().__init__()
        self.act_fn = F.relu
        self.h_dim = int(num_channels[0] * 4 * 4)
        self.first_conv_dim = (num_channels[0], 4, 4)
        self.fc1 = nn.Linear(latent_dim, self.h_dim)
        self.deconv_net = nn.Sequential(
            nn.ConvTranspose2d(num_channels[0], num_channels[1], (4, 4), stride=2, padding=1),
            #nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels[1], num_channels[2], (4, 4), stride=2, padding=1),
            #nn.BatchNorm2d(num_channels[2]),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels[2], num_channels[3], (4, 4), stride=2, padding=1),
            #nn.BatchNorm2d(num_channels[3]),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels[3], 1, (4, 4), stride=2, padding=1)
        )

    def forward(self, latent):
        h = self.act_fn(self.fc1(latent)).view(-1, *self.first_conv_dim)
        x = self.deconv_net(h)
        return x

if __name__ == '__main__':
    # This just about fits on a 11GB GPU (batch of 300 with training)
    m = Conv3DEncoder(512)
    m.cuda()
    x = torch.zeros(300, 1, 64, 64, 64, device='cuda:0')
    latent_mu, latent_std = m(x)
    latent = latent_mu + latent_std * torch.randn_like(latent_mu)

    d = Conv3DDecoder(512)
    d.cuda()
    x_hat = d(latent)

    l = (x_hat - x).sum()
    l.backward()
