import torch
from torch import nn
from flow_mpc.models.utils import CollisionFcn
from flow_mpc.models.generative_model import GenerativeModel

class DoubleIntegratorDynamics(nn.Module):

    def __init__(self, dim=2):
        super(DoubleIntegratorDynamics, self).__init__()
        dt = 0.05

        if dim == 2:
            # Add viscous damping to A matrix
            self.register_buffer('A', torch.tensor([[1.0, 0.0, dt, 0.0],
                                                    [0.0, 1.0, 0.0, dt],
                                                    [0.0, 0.0, 0.95, 0.0],
                                                    [0.0, 0.0, 0.0, 0.95]]))

            self.register_buffer('B', torch.tensor([[0.0, 0.0],
                                                    [0.0, 0.0],
                                                    [dt, 0.0],
                                                    [0.0, dt]]))
        elif dim == 3:
            # Add viscous damping to A matrix
            self.register_buffer('A', torch.tensor([[1.0, 0.0, 0.0, dt, 0.0, 0.0],
                                                    [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                                                    [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                                                    [0.0, 0.0, 0.0, 0.95, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.95, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.95]
                                                    ]))

            self.register_buffer('B', torch.tensor([[0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0],
                                                    [dt, 0.0, 0.0],
                                                    [0.0, dt, 0.0],
                                                    [0.0, 0.0, dt]
                                                    ]))
        else:
            raise ValueError('dim must either be 2 or 3')


    def forward(self, state, action):
        return self.batched_dynamics(state, action)

    def batched_dynamics(self, state, action):
        u = action  # torch.clamp(action, min=-10, max=10)
        return (self.A @ state.unsqueeze(2) + self.B @ u.unsqueeze(2)).squeeze()


class DoubleIntegratorModel(GenerativeModel):

    def __init__(self, world_dim=2):
        assert world_dim == 2 or world_dim == 3
        self.dworld = world_dim
        dynamics = DoubleIntegratorDynamics(dim=world_dim)
        prior = torch.distributions.Normal(loc=0.0, scale=1.0)

        super().__init__(dynamics=dynamics, action_prior=prior, state_dim=2*world_dim, control_dim=world_dim)

    @staticmethod
    def state_to_configuration(state):
        # Converts a full state to a position for checking the SDF
        config, _ = torch.chunk(state, dim=-1, chunks=2)
        return config

    def goal_log_likelihood(self, state, goal):
        state_config = self.state_to_configuration(state)
        goal_config = self.state_to_configuration(goal)
        return -10 * torch.norm(state_config - goal_config, dim=-1)

    def collision_log_likelihood(self, state, sdf, sdf_grad):
        config = self.state_to_configuration(state)
        sdf_val = self.collision_fn.apply(config, sdf, sdf_grad)
        return torch.where(sdf_val < 0, -10e4 * torch.ones_like(sdf_val), torch.zeros_like(sdf_val))

def trajectory_kernel(X, goals):
    # We do kind of a time convoluted traj kernel
    B, T, N, d = X.shape
    lengthscale = 0.7
    squared_distance = sq_diff(X, X)  # will return batch_size x traj_length x num_samples x num_samples
    squared_distance2 = sq_diff(X[:, :-1], X[:, 1:])
    squared_distance3 = sq_diff(X[:, :-2], X[:, 2:])
    distance_to_goal = 1e-5 + torch.linalg.norm(X.reshape(-1, d) - goals[:, :2].repeat(1, T, 1).reshape(-1, 2),
                                                dim=1).reshape(B, T, N, 1).repeat(1, 1, 1, N)
    weighting = 0.5 * (distance_to_goal + distance_to_goal.permute(0, 1, 3, 2))
    weighting2 = 0.5 * (distance_to_goal[:, :-1] + distance_to_goal[:, 1:].permute(0, 1, 3, 2))
    weighting3 = 0.5 * (distance_to_goal[:, :-2] + distance_to_goal[:, 2:].permute(0, 1, 3, 2))

    K = torch.exp(-squared_distance / (0.5 * weighting)).sum(dim=1) + \
        torch.exp(-squared_distance2 / (0.10 * weighting2)).sum(dim=1) + \
        torch.exp(-squared_distance3 / (0.05 * weighting3)).sum(dim=1)
    return K.mean(dim=[1, 2])


def sq_diff(x, y):
    ''' Expects trajectories batch_size x time x num_samples x state_dim'''
    # G = torch.einsum('btij, btjk->btik', (x, y.transpose(2, 3)))
    # diagonals = torch.einsum('btii->bti', G)
    # D = diagonals.unsqueeze(-2) + diagonals.unsqueeze(-1) - 2 * G
    B, T, N, d = x.shape
    D = torch.cdist(x.reshape(B * T, N, d), y.reshape(B * T, N, d), p=2).reshape(B, T, N, N)
    return D
