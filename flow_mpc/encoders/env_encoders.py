import torch
import torchvision
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
        self.encoder = []
        self.encoder.append(nn.Conv2d(1, 32, 3, stride=1, padding=1))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Conv2d(32, 32, 3, stride=1, padding=1))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.MaxPool2d(2, 2))
        self.encoder.append(nn.Conv2d(32, 64, 3, stride=1, padding=1))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Conv2d(64, 64, 3, stride=1, padding=1))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.MaxPool2d(2, 2))
        self.encoder.append(nn.Flatten())
        self.encoder.append(nn.Linear(4*image_size[0]*image_size[1], 256))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(256, z_env_dim))
        self.encoder = nn.Sequential(*self.encoder)

        self.decoder_1 = []
        self.decoder_1.append(nn.Linear(z_env_dim, 256))
        self.decoder_1.append(nn.ReLU())
        self.decoder_1.append(nn.Linear(256, 4*image_size[0]*image_size[1]))
        self.decoder_1 = nn.Sequential(*self.decoder_1)

        self.decoder_2 = []
        self.decoder_2.append(nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
        self.decoder_2.append(nn.ReLU())
        self.decoder_2.append(nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1))
        self.decoder_2.append(nn.ReLU())
        self.decoder_2.append(nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1))
        self.decoder_2.append(nn.ReLU())
        self.decoder_2.append(nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1))
        self.decoder_2.append(nn.ReLU())
        self.decoder_2.append(nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1))
        self.decoder_2.append(nn.ReLU())
        self.decoder_2.append(nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1))
        self.decoder_2 = nn.Sequential(*self.decoder_2)

    def encode(self, image, angle=None):
        """
        Encode the image. If angle is not None, the image after CNN layers will be rotated by the given angle before fed into the rest MLP
        :param image: a tensor of shape (B/N, channel, height, width)
        :param angle: a tensor of shape (B,)
        :return image_context: a tensor of shape (B/N, env_dim) or (B, env_dim), depending on whether angle is None
        """
        if angle is None:
            image_context = self.encoder(image)
        else:
            assert type(self.encoder[10]) is nn.Flatten
            angle = angle / torch.pi * 180
            image_context = self.encoder[0:10](image)
            N = angle.shape[0] // image.shape[0]
            image_context = image_context.unsqueeze(0).repeat(N, 1, 1, 1, 1).transpose(0, 1).reshape(-1, *image_context.shape[1:])
            new_image_context = image_context.clone()
            for i, single_image_context in enumerate(image_context):
                new_image_context[i] = torchvision.transforms.functional.rotate(single_image_context, angle[i].item())
            image_context = self.encoder[10:](new_image_context)
        return image_context

    def reconstruct(self, z):
        image = self.decoder_1(z)
        image = image.reshape(-1, 64, self.image_size[0]//4, self.image_size[1]//4)
        image = self.decoder_2(image)
        return image


class AttentionEncoder(nn.Module):
    # An attention architecture from the paper Attention is All You Need
    def __init__(self, context_dim, z_env_dim=64, image_latent_dim=64, num_blocks=3, num_heads=8):
        super(AttentionEncoder, self).__init__()
        self.z_env_dim = z_env_dim
        # image feature extractor that scales the image 4x down
        self.image_embedding = []
        self.image_embedding.append(nn.Conv2d(1, 32, 3, stride=1, padding=1))
        self.image_embedding.append(nn.ReLU())
        self.image_embedding.append(nn.Conv2d(32, 32, 3, stride=1, padding=1))
        self.image_embedding.append(nn.ReLU())
        self.image_embedding.append(nn.MaxPool2d(2, 2))
        self.image_embedding.append(nn.Conv2d(32, 64, 3, stride=1, padding=1))
        self.image_embedding.append(nn.ReLU())
        self.image_embedding.append(nn.Conv2d(64, 64, 3, stride=1, padding=1))
        self.image_embedding.append(nn.ReLU())
        self.image_embedding.append(nn.MaxPool2d(2, 2))
        self.image_embedding.append(nn.Conv2d(64, 128, 3, stride=1, padding=1))
        self.image_embedding.append(nn.ReLU())
        self.image_embedding.append(nn.Conv2d(128, 128, 3, stride=1, padding=1))
        self.image_embedding.append(nn.ReLU())
        self.image_embedding.append(nn.MaxPool2d(2, 2))
        self.image_embedding.append(nn.Conv2d(128, 128, 2, stride=2, padding=0))
        self.image_embedding.append(nn.Flatten(2, 3))
        self.image_embedding = nn.Sequential(*self.image_embedding)
        self.image_embedding_2 = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, image_latent_dim))
        # context projection
        self.context_project = nn.Linear(context_dim, z_env_dim)
        # cross attention block
        cross_attention_blocks = []
        for _ in range(num_blocks):
            cross_attention_blocks.append(nn.MultiheadAttention(embed_dim=z_env_dim, num_heads=num_heads, kdim=image_latent_dim, vdim=image_latent_dim, batch_first=True))
        self.cross_attention_blocks = nn.ModuleList(cross_attention_blocks)

        # # query, key and value matrix
        # self.q = nn.Linear(context_dim, z_env_dim)
        # self.k = nn.Linear(64, z_env_dim)
        # self.v = nn.Linear(64, z_env_dim)
        # # weight initialization
        # cq = (6 / (context_dim + z_env_dim)) ** 0.5
        # cv = (6 / (64 + z_env_dim)) ** 0.5
        # torch.nn.init.uniform_(self.q.weight, -cq, cq)
        # torch.nn.init.uniform_(self.k.weight, -cv, cv)
        # torch.nn.init.uniform_(self.v.weight, -cv, cv)

    def encode(self, image: torch.Tensor, context: torch.Tensor):
        """
        Given image and context (query), output the environment code for flow model and the corresponding attention mask
        :param image: grey image of shape (B/N, 1, H, W)
        :param context: context vector of shape (B, context_dim)
        :return env_code: environment code of shape (B, z_env_dim)
        :return attn_mask: attention mask of shape (B, H/16, W/16)
        """
        E, C, H, W = image.shape
        B = context.shape[0]
        N = B // E
        # image feature extraction
        image_feature = self.image_embedding(image).transpose(1, 2)     # shape (B/N, H*W/256, 64)
        image_feature = self.image_embedding_2(image_feature)           # shape (B/N, H*W/256, image_latent_dim)
        # cross attention (official)
        _, WH, image_latent_dim = image_feature.shape
        image_feature = image_feature.unsqueeze(0).repeat(N, 1, 1, 1).transpose(0, 1).reshape(-1, WH, image_latent_dim)  # shape (B, H*W/16, image_latent_dim)
        z = self.context_project(context).unsqueeze(1)
        attn_mask = image.new_zeros(B, 1, WH)
        for block in self.cross_attention_blocks:
            z, attn_mask_tmp = block(query=z, key=image_feature, value=image_feature)
            attn_mask = attn_mask + attn_mask_tmp
        env_code = z.squeeze(1)
        attn_mask = (attn_mask / len(self.cross_attention_blocks)).reshape(B, H//16, W//16)
        # # cross attention (customize)
        # WH = image_feature.shape[1]
        # # image_feature = image_feature.unsqueeze(0).repeat(N, 1, 1, 1).transpose(0, 1).reshape(-1, WH, 64)  # shape (B, H*W/16, z_env_dim)
        # key = self.k(image_feature)     # shape (B/N, H*W/16, z_env_dim)
        # value = self.v(image_feature)   # shape (B/N, H*W/16, z_env_dim)
        # key = key.unsqueeze(0).repeat(N, 1, 1, 1).transpose(0, 1).reshape(-1, WH, self.z_env_dim)  # shape (B, H*W/16, z_env_dim)
        # value = value.unsqueeze(0).repeat(N, 1, 1, 1).transpose(0, 1).reshape(-1, WH, self.z_env_dim)  # shape (B, H*W/16, z_env_dim)
        # query = self.q(context).unsqueeze(-1)         # shape (B, z_env_dim, 1)
        # weight = (key @ query) / self.z_env_dim**0.5         # shape (B, H*W/16, 1)
        # weight = torch.softmax(weight, dim=1)
        # env_code = (value.transpose(1, 2) @ weight).squeeze(-1)
        # attn_mask = weight.squeeze(-1).reshape(context.shape[0], H//4, W//4)
        return env_code, attn_mask



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