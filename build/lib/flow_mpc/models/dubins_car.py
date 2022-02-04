import torch
from torch import nn
from flow_mpc.models.utils import CollisionFcn
from flow_mpc.models.generative_model import GenerativeModel

class DubinsCarDynamics(nn.Module):

    def __init__(self):
        super(DubinsCarDynamics, self).__init__()
        self.dt = 0.1

    def forward(self, state, action):
        x, y, theta = torch.chunk(state, dim=-1, chunks=3)
        x_dot = torch.cos(theta)
        y_dot = torch.sin(theta)

        x = x + self.dt * x_dot
        y = y + self.dt * y_dot
        theta = theta + self.dt * action

        return torch.cat((x, y, theta), dim=-1)

class DubinsCarModel(GenerativeModel):

    def __init__(self, world_dim=2):
        assert world_dim == 2
        self.dworld = world_dim
        dynamics = DubinsCarDynamics()
        prior = torch.distributions.Normal(loc=0.0, scale=1.0)

        super().__init__(dynamics=dynamics, action_prior=prior, state_dim=3, control_dim=1)

    @staticmethod
    def state_to_configuration(state):
        # Converts a full state to a position for checking the SDF
        x, y, _ = torch.chunk(state, dim=-1, chunks=3)
        return torch.cat((x, y), dim=-1)

    def goal_log_likelihood(self, state, goal):
        return -torch.norm(state - goal, dim=-1)

    def collision_log_likelihood(self, state, sdf, sdf_grad):
        config = self.state_to_configuration(state)
        return self.collision_fn.apply(config, sdf, sdf_grad)
