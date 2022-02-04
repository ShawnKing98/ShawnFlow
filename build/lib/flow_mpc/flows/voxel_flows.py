# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html
import torch
from torch import nn
from torch.nn import functional as F

from .sequential import SequentialFlow
from .utils import create_channel_mask, create_half_mask
from .splits_and_priors import SplitFlow
from .splits_and_priors import SqueezeVoxelFlow as SqueezeFlow


class ConcatELU(nn.Module):
    """
    Activation function that applies ELU in both direction (inverted and plain).
    Allows non-linearity while providing strong gradients for any input (important for final convolution)
    """

    def forward(self, x):
        return torch.cat([F.elu(x), F.elu(-x)], dim=1)


class LayerNormChannels(nn.Module):

    def __init__(self, c_in):
        """
        This module applies layer norm across channels in an image. Has been shown to work well with ResNet connections.
        Inputs:
            c_in - Number of channels of the input
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_in)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class GatedConv(nn.Module):

    def __init__(self, c_in, c_hidden):
        """
        This module applies a two-layer convolutional ResNet block with input gate
        Inputs:
            c_in - Number of channels of the input
            c_hidden - Number of hidden dimensions we want to model (usually similar to c_in)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c_in, c_hidden, kernel_size=(3, 3, 3), padding=1),
            ConcatELU(),
            nn.Conv3d(2 * c_hidden, 2 * c_in, kernel_size=(1, 1, 1))
        )

    def forward(self, x):
        out = self.net(x)
        val, gate = out.chunk(2, dim=1)
        return x + val * torch.sigmoid(gate)


class GatedConvNet(nn.Module):

    def __init__(self, c_in, c_hidden=32, c_out=-1, num_layers=3):
        """
        Module that summarizes the previous blocks to a full convolutional neural network.
        Inputs:
            c_in - Number of input channels
            c_hidden - Number of hidden dimensions to use within the network
            c_out - Number of output channels. If -1, 2 times the input channels are used (affine coupling)
            num_layers - Number of gated ResNet blocks to apply
        """
        super().__init__()
        c_out = c_out if c_out > 0 else 2 * c_in
        layers = []
        layers += [nn.Conv3d(c_in, c_hidden, kernel_size=(3, 3, 3), padding=1)]
        for layer_index in range(num_layers):
            layers += [GatedConv(c_hidden, c_hidden),
                       LayerNormChannels(c_hidden)]
        layers += [ConcatELU(),
                   nn.Conv3d(2 * c_hidden, c_out, kernel_size=(3, 3, 3), padding=1)]
        self.nn = nn.Sequential(*layers)

        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, x):
        return self.nn(x)


class CouplingLayer(nn.Module):

    def __init__(self, network, mask, c_in):
        """
        Coupling layer inside a normalizing flow.
        Inputs:
            network - A PyTorch nn.Module constituting the deep neural network for mu and sigma.
                      Output shape should be twice the channel size as the input.
            mask - Binary mask (0 or 1) where 0 denotes that the element should be transformed,
                   while 1 means the latent will be used as input to the NN.
            c_in - Number of input channels
        """
        super().__init__()
        self.network = network
        self.scaling_factor = nn.Parameter(torch.zeros(c_in))
        # Register mask as buffer as it is a tensor which is not a parameter,
        # but should be part of the modules state.
        self.register_buffer('mask', mask)

    def forward(self, z, logpx, reverse=False, orig_img=None):
        """
        Inputs:
            z - Latent input to the flow
            logpx - The current logpx of the previous flows.
                  The logpx of this layer will be added to this tensor.
            reverse - If True, we apply the inverse of the layer.
            orig_img (optional) - Only needed in VarDeq. Allows external
                                  input to condition the flow on (e.g. original image)
        """
        # Apply network to masked input
        z_in = z * self.mask
        if orig_img is None:
            nn_out = self.network(z_in)
        else:
            nn_out = self.network(torch.cat([z_in, orig_img], dim=1))

        s, t = nn_out.chunk(2, dim=1)

        # Stabilize scaling output
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1, 1)
        s = torch.tanh(s / s_fac) * s_fac

        # Mask outputs (only transform the second part)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # Affine transformation
        if not reverse:
            z = z * torch.exp(s) + t
        else:
            z = (z - t) * torch.exp(-1 * s)

        return z, logpx + s.sum(dim=[1, 2, 3, 4])


class VoxelFlow(nn.Module):

    def __init__(self, imsize=(64, 64, 64)):
        super().__init__()
        self.imsize = imsize

        flow_chain = []
        # self.prior_size = (1, *imsize)

        for i in range(2):
            flow_chain.append(CouplingLayer(network=GatedConvNet(c_in=1, c_hidden=32),
                                            mask=create_half_mask(d=self.imsize[0], h=self.imsize[1], w=self.imsize[2],
                                                                  invert=(i % 2 == 1)),
                                            c_in=1))

        # Squeezes 1 x 64 x 64 x 64 -> 8 x 32 x 32 x 32
        flow_chain.append(SqueezeFlow())
        # split to 4 x 32 x 32 x 32
        flow_chain.append(SplitFlow())
        for i in range(2):
            flow_chain.append(CouplingLayer(network=GatedConvNet(c_in=4, c_hidden=48),
                                            mask=create_channel_mask(c_in=4,
                                                                     invert=(i % 2 == 1)).unsqueeze(-1),
                                            c_in=4))

        # Squeezes 4 x 32 x 32 x 32 -> 32 x 16 x 16 x 16
        flow_chain.append(SqueezeFlow())
        # Splits to 16 x 16 x 16 x 16
        flow_chain.append(SplitFlow())

        for i in range(4):
            flow_chain.append(CouplingLayer(network=GatedConvNet(c_in=16, c_hidden=64),
                                            mask=create_channel_mask(c_in=16, invert=(i % 2 == 1)).unsqueeze(-1),
                                            c_in=16))

        ## splits to 4 x 16 x 16
        # flow_chain.append(SplitFlow())

        # for i in range(2):
        #    flow_chain.append(CouplingLayer(network=GatedConvNet(c_in=4, c_hidden=64),
        #                                    mask=create_channel_mask(invert=(i % 2 == 1), c_in=4),
        #                                    c_in=4))

        # splits to 2 x 16 x 16
        # flow_chain.append(SplitFlow())

        # for i in range(2):
        #    flow_chain.append(CouplingLayer(network=GatedConvNet(c_in=2, c_hidden=64),
        #                                    mask=create_channel_mask(invert=(i % 2 == 1), c_in=2),
        #                                    c_in=2#))

        self.flow = SequentialFlow(flow_chain)

        # Prior is unit Gaussian
        self.prior = torch.distributions.Normal(loc=0, scale=1.0)
        self.prior_shape = (16, 16, 16, 16)
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        z, logpx = x, torch.zeros(x.shape[0], device=x.device)
        z, logpx = self.flow(z, logpx=logpx, context=None, reverse=False)
        logpx += self.prior.log_prob(z).sum(dim=[1, 2, 3, 4])
        return z, logpx

    def sample(self, N):
        z = self.prior.rsample(sample_shape=(N, *self.prior_shape)).to(device=self.dummy_param.device)
        logpx = 0
        x, logpx = self.flow(z, logpx=logpx, context=None, reverse=True)
        return x, logpx

    def reconstruct(self, z):
        logpx = torch.zeros(1, device=z.device)
        x, logpx = self.flow(z, logpx=logpx, context=None, reverse=True)
        return x, logpx
