import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor


class MaskLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, mask=None):
        super(MaskLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.register_buffer('mask', mask)

    def forward(self, input: Tensor, mask: Tensor = None) -> Tensor:
        if mask is None:
            mask = self.mask
            if mask is None:
                mask = 1
        return F.linear(input, self.weight * mask, self.bias)


class DoubleHeadConditionalMaskedAutoregressive(nn.Module):
    """
    A masked autoencoder following the paper 'MADE: Masked Autoencoder for Distribution Estimation'
    """
    def __init__(self, flow_dim, flow_order: torch.Tensor, context_order: torch.Tensor, hidden_dim=128, hidden_layer_num=2, nonlinearity='ReLu'):
        nn.Module.__init__(self)
        if nonlinearity == 'ReLu':
            self.nonlinearity = nn.ReLU(inplace=True)
        elif nonlinearity == 'Tanh':
            self.nonlinearity = nn.Tanh()
        max_order = max(flow_order.max(), context_order.max()).item()
        assert hidden_dim >= max_order-1       # hidden dim must be huge enough to convey the information needed
        context_dim = len(context_order)
        layers = [MaskLinear(flow_dim+context_dim, hidden_dim)]
        element_orders = [torch.cat((flow_order, context_order)),
                          torch.cat([torch.randperm(max_order-1) for _ in range(hidden_dim//(max_order-1))] + [torch.randperm(hidden_dim%(max_order-1))]),     # to make sure every order number occurs
                          ]
        for _ in range(hidden_layer_num-1):
            layers.append(MaskLinear(hidden_dim, hidden_dim))
            element_orders.append(torch.cat([torch.randperm(max_order-1) for _ in range(hidden_dim//(max_order-1))] + [torch.randperm(hidden_dim%(max_order-1))]))
        self.layers = nn.Sequential(*layers)
        # self.masks = []
        for i, layer in enumerate(layers):
            in_order = element_orders[i].repeat(layer.out_features, 1)
            out_order = element_orders[i+1].unsqueeze(1).repeat(1, layer.in_features)
            mask = out_order >= in_order
            self.register_buffer(f'mask_{i}', mask)
            # self.masks.append(mask)
        self.mu_layer = MaskLinear(hidden_dim, flow_dim)
        self.logvar_layer = MaskLinear(hidden_dim, flow_dim)
        in_order = element_orders[-1].repeat(flow_dim, 1)
        out_order = flow_order.unsqueeze(1).repeat(1, hidden_dim)
        self.register_buffer('mask_out', out_order > in_order)

    def forward(self, x, context):
        """
        Take in x and context, output the mean and log variance of a distribution
        :param x: input tensor with shape (N, Dx)
        :param context: context tensor with shape (N, Dc)
        :return mu: mean tensor with shape (N, Dx)
        :return logvar: log variance tensor with shape (N, Dx)
        """
        out = torch.cat((x, context), dim=-1)
        for i, layer in enumerate(self.layers):
            out = layer(out, getattr(self, f'mask_{i}'))
            out = self.nonlinearity(out)
        mu = self.mu_layer(out, self.mask_out)
        logvar = self.logvar_layer(out, self.mask_out)
        return mu, logvar


class MaskedAutoRegressiveLayer(nn.Module):
    """Used in 2D experiments."""

    def __init__(self, horizon, channel, context_order: torch.Tensor, intermediate_dim=128, nonlinearity='ReLu'):
        nn.Module.__init__(self)
        self.horizon = horizon
        self.channel = channel
        flow_order = torch.arange(horizon).repeat(channel, 1).T.reshape(-1)
        self.scaling_factor = nn.Parameter(torch.zeros(horizon*channel))
        self.net = DoubleHeadConditionalMaskedAutoregressive(horizon*channel, flow_order, context_order, hidden_dim=intermediate_dim, nonlinearity=nonlinearity)

    def forward(self, x: torch.Tensor, context: torch.Tensor, logpx=None, reverse=False):
        """
        Take in a variable with shape (B, Dx) and output a variable with the same shape & the probability, conditioning on a context variable with shape (B, Dc)
        :param x: input to the layer, could be the latent variable, OR could be the original variable, depending on param reverse
        :param context: the context variable used to condition the output distribution
        :param logpx: log probability before this layer
        :param reverse: True means from original space to latent space, False means from latent space to original space
        :return y: the output original variable OR the output latent variable
        :return logpy: log probability after this layer, a B-dim tensor
        """
        if not reverse:         # sampling
            y = torch.zeros_like(x)
            for i in range(self.horizon):
                mu, logvar = self.net(y, context)
                s_fac = self.scaling_factor.exp()
                scale = torch.tanh(logvar / s_fac) * s_fac
                y = (x - mu) * (-scale).exp()
            ldj = -scale.sum(dim=1)        # dx/dz
        else:                   # training
            mu, logvar = self.net(x, context)
            s_fac = self.scaling_factor.exp()
            scale = torch.tanh(logvar / s_fac) * s_fac
            y = x * scale.exp() + mu
            ldj = scale.sum(dim=1)         # dz/dx
        if logpx is None:
            return y
        else:
            return y, logpx + ldj
