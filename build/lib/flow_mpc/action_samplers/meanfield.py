import torch
from torch import nn
from .base import BaseActionSampler
from torch.distributions.normal import Normal

class MeanFieldVIActionSampler(BaseActionSampler):


    def __init__(self, context_net, environment_encoder, action_dimension, horizon, hidden_size=256):
        super().__init__(context_net, environment_encoder, action_dimension, horizon)

        self.net = nn.Sequential(
            nn.Linear(context_net.context_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.du * self.H * 2)
        )
        self.softplus = nn.Softplus()

    def _get_action_distribution(self, start, goal, environment):
        context = self.context_net(start, goal, environment)
        u_mu, u_std = torch.chunk(self.net(context), dim=1, chunks=2)
        u_std = self.softplus(u_std)
        return Normal(u_mu, u_std)

    def sample(self, start, goal, environment, N=1):
        qu = self._get_action_distribution(start, goal, environment)
        u = qu.rsample(sample_shape=(N,))
        return u

    def likelihood(self, u, start, goal, environment):
        qu = self._get_action_distribution(start, goal, environment)
        log_qu = qu.log_prob(u)
        return log_qu
