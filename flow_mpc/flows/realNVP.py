import torch
from torch import nn
from flow_mpc import flows


class RealNVPModel(nn.Module):
    def __init__(self, state_dim, action_dim, horizon):
        super(RealNVPModel, self).__init__()
        self.flow = self.build_nvp_flow(state_dim * horizon, state_dim + action_dim * horizon)
        self.conditional_prior = flows.ConditionalPrior(state_dim + action_dim * horizon, state_dim * horizon)

    @staticmethod
    def build_nvp_flow(flow_dim, context_dim, flow_length=8):
        """
        Build a flow
        :param flow_dim: the dimension of variables to be flowed
        :param context_dim: the dimension of the conditioning variables
        :param flow_length: the number of the stacked flow layers
        :return: a flow
        """
        flow_list = []
        for _ in range(flow_length):
            flow_list.append(flows.CouplingLayer(flow_dim, context_dim=context_dim, intermediate_dim=64))
            # flow_list.append(flows.BatchNormFlow(flow_dim))
            # flow_list.append(flows.LULinear(flow_dim, context_dim=context_dim))

            # flow_list.append(flows.LULinear(flow_dim))
            flow_list.append(flows.RandomPermutation(features=flow_dim))
        flow_list.append(flows.CouplingLayer(flow_dim, context_dim=context_dim, intermediate_dim=64))
        return flows.SequentialFlow(flow_list)

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
        :return: predicted_traj of shape (B, horizon, state_dim) OR latent variable of shape (B, horizon x state_dim) depending on reverse or not, and a log prob
        """
        batch_size = start_state.shape[0]
        state_dim = start_state.shape[1]
        context = torch.cat((start_state, action.reshape(batch_size, -1)), dim=1)
        if not reverse: # sampling
            z, log_prob = self.conditional_prior(z=None, logpx=0, context=context, reverse=reverse)
            x, ldj = self.flow(z, logpx=0, context=context, reverse=reverse)
            return x.reshape(batch_size, -1, state_dim), log_prob + ldj
        else:
            x = traj.reshape(batch_size, -1)
            z, ldj = self.flow(x, logpx=0, context=context, reverse=reverse)
            z, log_prob = self.conditional_prior(z=z, logpx=ldj, context=context, reverse=reverse)
            return z, log_prob
