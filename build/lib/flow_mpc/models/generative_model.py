import torch
from torch import nn
from flow_mpc.models.utils import PointGoalFcn, CollisionFcn


class GenerativeModel(nn.Module):

    def __init__(self, dynamics, action_prior, state_dim, control_dim):
        super().__init__()
        self.dynamics = dynamics
        self.dynamics_sigma = 0.00
        self.collision_fn = CollisionFcn()
        self.dx = state_dim
        self.du = control_dim
        self.prior = action_prior

    def goal_log_likelihood(self, state, goal):
        raise NotImplementedError

    def collision_log_likelihood(self, state, sdf, sdf_grad):
        raise NotImplementedError

    def forward(self, start, goal, sdf, sdf_grad, action_sequence):
        B, N, dx = start.shape
        _, _, H, du = action_sequence.shape

        assert dx == self.dx
        assert du == self.du

        action_logprob = self.prior.log_prob(action_sequence).sum(dim=[-2, -1])# / du
        goal_logprob = 0
        collision_logprob = 0
        x_sequence = []
        x = start
        for t in range(H):
            action = action_sequence[:, :, t]
            x_mu = self.dynamics(x.reshape(-1, dx), action.reshape(-1, du)).reshape(B, N, dx)
            x = x_mu + self.dynamics_sigma * torch.randn_like(x_mu)
            x_sequence.append(x_mu)
            goal_logprob += self.goal_log_likelihood(x, goal)

        goal_logprob += 10 * self.goal_log_likelihood(x, goal)
        X = torch.stack(x_sequence, dim=2) # B x N x T x dx
        # do all collision in one go to save time ( i hope)
        collision_logprob = self.collision_log_likelihood(X.reshape(B, -1, dx),
                                                          sdf, sdf_grad).reshape(B, N, H).sum(dim=2)

        cost_logprob = collision_logprob + goal_logprob
        cost_logprob = torch.where(
            torch.logical_or(torch.isnan(cost_logprob), torch.isinf(cost_logprob)),
            -1e7 * torch.ones_like(cost_logprob),
            cost_logprob
        )

        if N == 1:
            print(collision_logprob, goal_logprob)
        return cost_logprob, action_logprob, X
