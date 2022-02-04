import torch
from torch import nn
from torch.nn import functional as F


class BaseContextNet(nn.Module):
    """
    Base for conditioning nets

    Conditioning nets preprocess the information on which the action sampler is conditioned

    This information is the start, the goal, and the environments

    """

    def __init__(self):
        super().__init__()

    def forward(self, start, goal, environment):
        raise NotImplementedError


class ConditioningNetwork(nn.Module):

    def __init__(self, context_dim, z_env_dim, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_env_dim + 2 * state_dim, context_dim)
        self.fc2 = nn.Linear(context_dim, context_dim)
        self.act_fn = F.relu
        self.context_dim = context_dim

    def forward(self, start, goal, z_env):
        h = torch.cat((start, goal, z_env), dim=1)
        context = self.act_fn(self.fc1(h))
        context = self.fc2(context) + context
        return context
