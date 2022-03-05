import torch
from torch import nn

class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.9, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, x, context=None, logpx=None, reverse=False):
        if self.training and reverse:
            self.batch_mean = x.mean(0)
            self.batch_var = (x - self.batch_mean).pow(2).mean(0) + self.eps

            self.running_mean.mul_(self.momentum)
            self.running_var.mul_(self.momentum)

            self.running_mean.add_(self.batch_mean.data *
                                   (1 - self.momentum))
            self.running_var.add_(self.batch_var.data *
                                  (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        if reverse:     # train
            x_hat = (x - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            ldj = (self.log_gamma - 0.5 * torch.log(var)).sum(-1)
        else:           # test
            x_hat = (x - self.beta) / torch.exp(self.log_gamma)
            y = x_hat * var.sqrt() + mean
            ldj = -(self.log_gamma - 0.5 * torch.log(var)).sum(-1)
        if logpx is None:
            return y
        else:
            return y, logpx + ldj