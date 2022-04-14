import ipdb
import torch
from torch import nn
from flow_mpc import flows
from flow_mpc.encoders import Encoder
from typing import List, Tuple


class RealNVPFlow(nn.Module):
    """The RealNVP flow"""
    def __init__(self, flow_dim, context_dim, flow_length, hidden_dim, initialized=False):
        """
                Build a RealNVP flow
                :param flow_dim: the dimension of variables to be flowed
                :param context_dim: the dimension of the conditioning variables
                :param flow_length: the number of the stacked flow layers
                :return: a flow
                """
        super(RealNVPFlow, self).__init__()
        flow_list = []
        for _ in range(flow_length):
            # flow_list.append(flows.CouplingLayer(flow_dim, context_dim=context_dim, intermediate_dim=hidden_dim))
            flow_list.append(flows.ResNetCouplingLayer(flow_dim, context_dim=context_dim, intermediate_dim=hidden_dim))
            flow_list.append(flows.ActNorm(flow_dim, initialized=initialized))
            # flow_list.append(flows.BatchNormFlow(flow_dim))
            # flow_list.append(flows.LULinear(flow_dim, context_dim=context_dim))

            # flow_list.append(flows.LULinear(flow_dim))
            flow_list.append(flows.RandomPermutation(features=flow_dim))
        if flow_length > 0:
            flow_list.append(flows.ResNetCouplingLayer(flow_dim, context_dim=context_dim, intermediate_dim=hidden_dim))
        self.flow = flows.SequentialFlow(flow_list)


class RealNVPModel(RealNVPFlow):
    """The dynamic model based on RealNVP"""
    def __init__(self, state_dim, action_dim, horizon, hidden_dim=256, flow_length=10, condition=True, initialized=False):
        super(RealNVPModel, self).__init__(flow_dim=state_dim * horizon,
                                           context_dim=state_dim + action_dim * horizon,
                                           flow_length=flow_length, hidden_dim=hidden_dim, initialized=initialized)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon

        if condition:
            self.prior = flows.ConditionalPrior(state_dim + action_dim * horizon, state_dim * horizon, hidden_dim=hidden_dim)
        else:
            self.prior = flows.GaussianPrior(state_dim * horizon)

    # @staticmethod
    # def build_nvp_flow(flow_dim, context_dim, flow_length, hidden_dim, initialized=False):
    #     """
    #     Build a RealNVP flow
    #     :param flow_dim: the dimension of variables to be flowed
    #     :param context_dim: the dimension of the conditioning variables
    #     :param flow_length: the number of the stacked flow layers
    #     :return: a flow
    #     """
    #     flow_list = []
    #     for _ in range(flow_length):
    #         # flow_list.append(flows.CouplingLayer(flow_dim, context_dim=context_dim, intermediate_dim=hidden_dim))
    #         flow_list.append(flows.ResNetCouplingLayer(flow_dim, context_dim=context_dim, intermediate_dim=hidden_dim))
    #         flow_list.append(flows.ActNorm(flow_dim, initialized=initialized))
    #         # flow_list.append(flows.BatchNormFlow(flow_dim))
    #         # flow_list.append(flows.LULinear(flow_dim, context_dim=context_dim))
    #
    #         # flow_list.append(flows.LULinear(flow_dim))
    #         flow_list.append(flows.RandomPermutation(features=flow_dim))
    #     if flow_length > 0:
    #         flow_list.append(flows.ResNetCouplingLayer(flow_dim, context_dim=context_dim, intermediate_dim=hidden_dim))
    #     # flow_list.append(flows.CouplingLayer(flow_dim, context_dim=context_dim, intermediate_dim=hidden_dim))
    #     return flows.SequentialFlow(flow_list)

    def forward(self, start_state: torch.Tensor, action: torch.Tensor, reverse=False, traj=None):
        """
        Forward:
        Given state at start time and actions along a time horizon, sample a predicted state trajectory and return its log prob
        Reverse:
        Given state at start time, actions along a time horizon, and a state trajectory, return the corresponding latent variable and its log_prob
        :param start_state: torch tensor of shape (B, state_dim)
        :param action: torch tensor of shape (B, horizon, action_dim)
        :param reverse: False means sampling and using this model, True means using passed-in trajectory to get the latent variable and its prob
        :param traj: torch tensor of shape (B, horizon, state_dim)
        :return predicted_traj of shape (B, horizon, state_dim) OR latent variable of shape (B, horizon x state_dim) depending on reverse or not
        :return log_prob: the probability of generating the predicted trajectory
        """
        batch_size = start_state.shape[0]
        context = torch.cat((start_state, action.reshape(batch_size, -1)), dim=1)
        if not reverse: # sampling
            z, log_prob = self.prior(z=None, logpx=0, context=context, reverse=reverse)
            x, ldj = self.flow(z, logpx=0, context=context, reverse=reverse)
            # x, ldj = z, 0
            relevant_replacement = x.reshape(batch_size, -1, self.state_dim)
            traj = start_state.unsqueeze(1) + torch.cumsum(relevant_replacement, dim=1)
            return traj, log_prob + ldj
        else:           # training
            before_traj = torch.cat((start_state.unsqueeze(1), traj[:, :-1]), dim=1)
            relevant_replacement = traj - before_traj
            x = relevant_replacement.reshape(batch_size, -1)
            z, ldj = self.flow(x, logpx=0, context=context, reverse=reverse)
            # z, ldj = x, 0
            z, log_prob = self.prior(z=z, logpx=ldj, context=context, reverse=reverse)
            return z, log_prob

class ImageRealNVPModel(RealNVPFlow):
    """The dynamic model based on RealNVP, along with an extra encoder to encode environment image"""
    def __init__(self, state_dim, action_dim, horizon, image_size: Tuple[int, int], env_dim=64, hidden_dim=256, flow_length=10, condition=True, initialized=False,
                 state_mean=(0, 0, 0, 0), state_std=(1, 1, 1, 1),
                 action_mean=(0, 0), action_std=(1, 1),
                 image_mean=torch.zeros(()), image_std=torch.ones(()),
                 ):
        super(ImageRealNVPModel, self).__init__(flow_dim=state_dim * horizon,
                                                context_dim=env_dim + env_dim,
                                                flow_length=flow_length, hidden_dim=hidden_dim, initialized=initialized)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.image_size = image_size
        self.env_dim = env_dim
        self.register_buffer('state_mean', torch.tensor(state_mean, dtype=torch.float))
        self.register_buffer('state_std', torch.tensor(state_std, dtype=torch.float))
        self.register_buffer('action_mean', torch.tensor(action_mean, dtype=torch.float))
        self.register_buffer('action_std', torch.tensor(action_std, dtype=torch.float))
        self.register_buffer('image_mean', torch.tensor(image_mean, dtype=torch.float))
        self.register_buffer('image_std', torch.tensor(image_std, dtype=torch.float))

        if condition:
            self.prior = flows.ConditionalPrior(env_dim + env_dim, state_dim * horizon, hidden_dim=hidden_dim)
        else:
            self.prior = flows.GaussianPrior(state_dim * horizon)

        self.encoder = Encoder(image_size, env_dim)
        self.s_u_encoder = nn.Sequential(nn.Linear(state_dim + action_dim * horizon, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, env_dim)
                                         )
        # self.s_u_encoder.requires_grad_(False)

    def forward(self, start_state: torch.Tensor, action: torch.Tensor, image: torch.Tensor, reverse=False, traj=None):
        """
        Forward:
        Given state at start time, actions along a time horizon, and an environment image, sample a predicted state trajectory and return its log prob
        Reverse:
        Given state at start time, actions along a time horizon, an environment image and a state trajectory, return the corresponding latent variable and its log_prob
        :param start_state: torch tensor of shape (B, state_dim)
        :param action: torch tensor of shape (B, horizon, action_dim)
        :param image: torch tensor of shape (B, channel, height, width)
        :param reverse: False means sampling and using this model, True means using passed-in trajectory to get the latent variable and its prob
        :param traj: torch tensor of shape (B, horizon, state_dim)
        :return: predicted_traj of shape (B, horizon, state_dim) OR latent variable of shape (B, horizon x state_dim) depending on reverse or not, and a log prob
        """
        start_state = (start_state - self.state_mean) / self.state_std
        action = (action - self.action_mean) / self.action_std
        image = (image - self.image_mean) / self.image_std
        batch_size = start_state.shape[0]
        env_code = self.encoder.encode(image)   # shape of (B, env_dim)
        s_u = torch.cat((start_state, action.reshape(batch_size, -1)), dim=1)
        s_u_code = self.s_u_encoder(s_u)
        context = torch.cat((s_u_code, env_code), dim=1)
        if not reverse: # sampling
            z, log_prob = self.prior(z=None, logpx=0, context=context, reverse=reverse)
            x, ldj = self.flow(z, logpx=0, context=context, reverse=reverse)
            relevant_replacement = x.reshape(batch_size, -1, self.state_dim)
            traj = start_state.unsqueeze(1) + torch.cumsum(relevant_replacement, dim=1)
            traj = traj * self.state_std + self.state_mean
            return traj, log_prob + ldj
        else:           # training
            assert traj is not None
            traj = (traj - self.state_mean) / self.state_std
            before_traj = torch.cat((start_state.unsqueeze(1), traj[:, :-1]), dim=1)
            relevant_replacement = traj - before_traj
            x = relevant_replacement.reshape(batch_size, -1)
            z, ldj = self.flow(x, logpx=0, context=context, reverse=reverse)
            # z, ldj = x, 0
            z, log_prob = self.prior(z=z, logpx=ldj, context=context, reverse=reverse)
            return z, log_prob
