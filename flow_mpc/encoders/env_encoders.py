import torch
from torch import nn
from torch.nn import functional as F
from flow_mpc.flows.image_flows import ImageFlow
from flow_mpc.flows.voxel_flows import VoxelFlow
from flow_mpc.encoders.vae import VAE
from torch.distributions.normal import Normal
from typing import Tuple


class BaseEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def encode(self, environment):
        raise NotImplementedError

    def get_ood_score(self, environment):
        raise NotImplementedError

    def reconstruct(self, z_env):
        raise NotImplementedError


class Encoder(nn.Module):

    def __init__(self, image_size: Tuple[int, int], z_env_dim):
        super(Encoder, self).__init__()
        self.image_size = image_size
        # convolutons for environments image
        self.net = []
        self.net.append(nn.Conv2d(1, 32, 3, stride=1, padding=1))
        self.net.append(nn.ReLU())
        self.net.append(nn.Conv2d(32, 32, 3, stride=1, padding=1))
        self.net.append(nn.ReLU())
        self.net.append(nn.MaxPool2d(2, 2))
        self.net.append(nn.Conv2d(32, 64, 3, stride=1, padding=1))
        self.net.append(nn.ReLU())
        self.net.append(nn.Conv2d(64, 64, 3, stride=1, padding=1))
        self.net.append(nn.ReLU())
        self.net.append(nn.MaxPool2d(2, 2))
        self.net.append(nn.Flatten())
        self.net.append(nn.Linear(4*image_size[0]*image_size[1], 256))
        self.net.append(nn.ReLU())
        self.net.append(nn.Linear(256, z_env_dim))
        self.net = nn.Sequential(*self.net)

    def encode(self, image):
        return self.net(image)


class EnsembleEncoder(nn.Module):

    def __init__(self, context_dim, z_env_dim, num_ensembles):
        super().__init__()

        for _ in range(num_ensembles):
            self.nets.append(ConditioningNetwork(context_dim))

        self.nets = nn.ModuleList(self.nets)

    def encode(self, environment):
        h_env = []
        for net in self.nets:
            h_env.append(net(start, goal, environment))

        h_env = torch.stack(context, dim=0)

        raise NotImplementedError

    def get_ood_score(self, environment):
        raise NotImplementedError


class FlowEncoder(nn.Module):

    def __init__(self, context_dim, z_env_dim):
        super().__init__()
        self.flow = ImageFlow()
        self.prior = Normal(loc=0.0, scale=1.0)

        self.conv1 = nn.Conv2d(8, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.fc = nn.Linear(512, z_env_dim)
        self.act_fn = F.relu
        self.dummy_param = nn.Parameter(torch.empty(0))

    def encode(self, environment, z_env=None):
        if z_env is None:
            z_env, log_p_env = self.flow(environment)
        else:
            log_p_env = None

        h_env = self.act_fn(self.conv1(z_env))
        h_env = self.act_fn(self.conv2(h_env))
        h_env = self.act_fn(self.conv3(h_env))
        h_env = h_env.reshape(-1, 512)
        h_env = self.fc(h_env)
        out = {
            'h_environment': h_env,
            'z_environment': z_env,
            'log_p_env': log_p_env
        }
        return out

    def reconstruct(self, z_env, N=1):
        B, _, _, _ = z_env.shape
        log_p_z_env = self.prior.log_prob(z_env).sum(dim=[1, 2, 3])
        environment, log_p_env = self.flow.reconstruct(z_env.repeat(N, 1, 1, 1, 1).reshape(-1, *self.flow.prior_shape))
        out = {
            'environments': environment.reshape(N, B, 1, 64, 64),
            'log_p_env': log_p_env.reshape(N, B) + log_p_z_env.repeat(N, 1)
        }
        return out

    def sample(self, N=1):
        z_env = self.prior.sample(sample_shape=(N, *self.flow.prior_shape)).to(device=self.dummy_param.device)
        out = self.reconstruct(z_env, N=1)
        return {
            'environments': out['environments'].reshape(N, 1, 64, 64),
            'log_p_env': out['log_p_env'].reshape(N)
        }


class VoxelFlowEncoder(nn.Module):

    def __init__(self, context_dim, z_env_dim):
        super().__init__()
        self.flow = VoxelFlow()
        self.prior = Normal(loc=0.0, scale=1.0)

        self.conv1 = nn.Conv3d(16, 32, 3, stride=2)
        self.conv2 = nn.Conv3d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.fc = nn.Linear(2048, z_env_dim)
        self.act_fn = F.relu
        self.dummy_param = nn.Parameter(torch.empty(0))

    def encode(self, environment, z_env=None):
        if z_env is None:
            z_env, log_p_env = self.flow(environment)
        else:
            log_p_env = None

        h_env = self.act_fn(self.conv1(z_env))
        h_env = self.act_fn(self.conv2(h_env))
        h_env = self.act_fn(self.conv3(h_env))
        h_env = h_env.reshape(-1, 2048)
        h_env = self.fc(h_env)
        out = {
            'h_environment': h_env,
            'z_environment': z_env,
            'log_p_env': log_p_env
        }
        return out

    def reconstruct(self, z_env, N=1):
        B, _, _, _ = z_env.shape
        log_p_z_env = self.prior.log_prob(z_env).sum(dim=[1, 2, 3])
        environment, log_p_env = self.flow.reconstruct(z_env.repeat(N, 1, 1, 1, 1).reshape(-1, *self.flow.prior_shape))
        out = {
            'environments': environment.reshape(N, B, 1, 64, 64),
            'log_p_env': log_p_env.reshape(N, B) + log_p_z_env.repeat(N, 1)
        }
        return out

    def sample(self, N=1):
        z_env = self.prior.sample(sample_shape=(N, *self.flow.prior_shape)).to(device=self.dummy_param.device)
        out = self.reconstruct(z_env, N=1)
        return {
            'environments': out['environments'].reshape(N, 1, 64, 64),
            'log_p_env': out['log_p_env'].reshape(N)
        }


class VoxelCompressedFlowEncoder(nn.Module):

    def __init__(self, context_dim, z_env_dim):
        super().__init__()
        self.flow = ImageFlow()
        self.prior = Normal(loc=0.0, scale=1.0)

        self.conv1 = nn.Conv3d(1, 32, 5, stride=3)
        self.conv2 = nn.Conv3d(32, 32, 5, stride=2)
        self.conv3 = nn.Conv3d(32, 48, 3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(48, 64, 3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(8, 32, 3, stride=2)
        self.conv6 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.fc = nn.Linear(512, z_env_dim)

        self.act_fn = F.relu
        self.dummy_param = nn.Parameter(torch.empty(0))

    def encode(self, environment, z_env=None):
        h_env = self.act_fn(self.conv1(environment))
        h_env = self.act_fn(self.conv2(h_env))
        h_env = self.act_fn(self.conv3(h_env))
        h_env = self.act_fn(self.conv4(h_env))
        h_env = h_env.reshape(-1, 1, 64, 64)

        if z_env is None:
            z_env, log_p_env = self.flow(h_env)
        else:
            log_p_env = None

        h_env = self.act_fn(self.conv5(z_env))
        h_env = self.act_fn(self.conv6(h_env))
        h_env = self.act_fn(self.conv7(h_env))
        h_env = h_env.reshape(-1, 512)
        h_env = self.fc(h_env)
        out = {
            'h_environment': h_env,
            'z_environment': z_env,
            'log_p_env': log_p_env
        }
        return out

    def reconstruct(self, z_env, N=1):
        B, _, _, _ = z_env.shape
        log_p_z_env = self.prior.log_prob(z_env).sum(dim=[1, 2, 3])
        environment, log_p_env = self.flow.reconstruct(
            z_env.repeat(N, 1, 1, 1, 1).reshape(-1, *self.flow.prior_shape))
        out = {
            'environments': environment.reshape(N, B, 1, 64, 64),
            'log_p_env': log_p_env.reshape(N, B) + log_p_z_env.repeat(N, 1)
        }
        return out

    def sample(self, N=1):
        z_env = self.prior.sample(sample_shape=(N, *self.flow.prior_shape)).to(device=self.dummy_param.device)
        out = self.reconstruct(z_env, N=1)
        return {
            'environments': out['environments'].reshape(N, 1, 64, 64),
            'log_p_env': out['log_p_env'].reshape(N)
        }


class VAEEncoder(nn.Module):

    def __init__(self, context_dim, z_env_dim, voxels=False, flow_prior=None):
        super().__init__()
        self.z_env_dim = z_env_dim
        self.vae = VAE(z_env_dim, flow_prior=flow_prior, voxels=voxels)
        self.dummy_param = nn.Parameter(torch.empty(0))

    def encode(self, environment, z_env=None, reconstruct=True):
        if z_env is not None:
            return {
                'z_environment': z_env,
            }

        out = {}
        B = environment.shape[0]
        # Actually return variational lower bound to log_p_env
        if reconstruct:
            env_hat, z_env, latent_mu, latent_sigma = self.vae(environment)
            sq_diff = -(env_hat - environment)**2
            kl_term = self.vae.get_kl_divergence(z_env, latent_mu, latent_sigma)
            log_p_env = sq_diff.view(B, -1).sum(dim=1)  - kl_term
            out['log_p_env'] = log_p_env
        else:
            z_env, _, _ = self.vae.encode(environment)
            out['log_p_env'] = None

        out['z_environment'] = z_env
        return out

    def reconstruct(self, z_env, N=1):
        #if N > 1:
        #    raise NotImplementedError("can't sample N>1 for vae")
        B = z_env.shape[0]
        #environment = self.vae.decoder(z_env)
        environment = self.vae.decoder(z_env.repeat(1, N, 1).reshape(B*N, -1))

        out = {
            'environments': environment.reshape(N, B, *environment.shape[1:]),
        }
        return out

    def sample(self, N=1):
        latent = self.vae.prior.sample(sample_shape=(N, self.z_env_dim)).to(device=self.dummy_param.device)
        return {
            'environments': self.vae.decoder(latent)
        }