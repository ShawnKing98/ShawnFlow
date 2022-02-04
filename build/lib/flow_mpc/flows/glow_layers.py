import torch
from torch import nn


class ActNorm(nn.Module):
    """ An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_x):
        super(ActNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_x))
        self.bias = nn.Parameter(torch.zeros(num_x))
        self.initialized = False

    def forward(self, x, context=None, logpx=None, reverse=False):
        if self.initialized == False:
            self.weight.data.copy_(torch.log(1.0 / (x.std(0) + 1e-12)))
            self.bias.data.copy_(x.mean(0))
            self.initialized = True

        if not reverse:
            y = (x - self.bias) * torch.exp(self.weight)
        else:
            y = x * torch.exp(-self.weight) + self.bias

        if logpx is None:
            return y
        return y, logpx - self.weight.sum(-1).repeat(x.size(0))