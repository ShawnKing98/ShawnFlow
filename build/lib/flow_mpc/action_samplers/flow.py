import torch
from torch import nn
from .base import BaseActionSampler
from torch.distributions.normal import Normal

from flow_mpc.flows import ResNetCouplingLayer as CouplingLayer
from flow_mpc.flows import RandomPermutation, BatchNormFlow, ActNorm
from flow_mpc.flows import LULinear
from flow_mpc.flows.sequential import SequentialFlow
from flow_mpc.flows.splits_and_priors import ConditionalSplitFlow, ConditionalPrior

from flow_mpc.flows.ffjord import layers
from flow_mpc.flows.OTFlow.ot_flow import OTFlow

def set_cnf_options(solver, model):

    def _set(module):
        if isinstance(module, layers.CNF):
            # Set training settings
            module.solver = solver
            module.atol = 1e-5
            module.rtol = 1e-5

            # If using fixed-grid adams, restrict order to not be too high.
            if solver in ['fixed_adams', 'explicit_adams']:
                module.solver_options['max_order'] = 4
            module.solver_options['first_step'] = 0.2

            # Set the test settings
            module.test_solver = solver
            module.test_atol = 1e-5
            module.test_rtol = 1e-5

        if isinstance(module, layers.ODEfunc):
            module.rademacher = False
            module.residual = False

    model.apply(_set)

def build_ffjord(flow_dim, context_dim, flow_length):

    hidden_dims = (64, 64, 64)
    divergence_fn = 'approximate'
    layer_type = 'concatsquash'
    nonlinearity = 'tanh'
    solver = 'dopri5'
    bn_lag = 0
    batch_norm = False
    T = 0.5
    def build_cnf():
        diffeq = layers.ODEnet(
            hidden_dims=hidden_dims,
            input_dim=flow_dim,
            context_dim=context_dim,
            strides=None,
            conv=False,
            layer_type=layer_type,
            nonlinearity=nonlinearity,
        )
        odefunc = layers.RegularizedODEfunc(layers.ODEfunc(
            diffeq=diffeq,
            divergence_fn=divergence_fn,
            residual=False,
            rademacher=False,
        ))
        cnf = layers.CNF(
            odefunc=odefunc,
            T=T,
            train_T=True,
            regularization_fns=None,
            solver=solver,
        )
        return cnf

    chain = [build_cnf() for _ in range(flow_length)]
    if batch_norm:
        bn_layers = [layers.MovingBatchNorm1d(flow_dim, bn_lag=bn_lag) for _ in range(flow_length)]
        bn_chain = [layers.MovingBatchNorm1d(flow_dim, bn_lag=bn_lag)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = SequentialFlow(chain)

    set_cnf_options(solver, model)

    return model


def build_nvp_flow(flow_dim, context_dim, flow_length):
    flow_chain = []
    for _ in range(flow_length):
        flow_chain.append(CouplingLayer(flow_dim, context_dim=context_dim))
        flow_chain.append(BatchNormFlow(flow_dim))
        flow_chain.append(RandomPermutation(flow_dim))
        flow_chain.append(LULinear(flow_dim, context_dim=context_dim))
    flow_chain.append(CouplingLayer(flow_dim, context_dim=context_dim))
    return SequentialFlow(flow_chain)


class FlowActionSampler(BaseActionSampler):

    def __init__(self, context_net, environment_encoder, action_dimension, horizon, flow_length, flow_type='nvp'):
        super().__init__(context_net, environment_encoder, action_dimension, horizon)

        flow_dim = self.du * self.H
        if flow_type == 'ffjord':
            self.flow = build_ffjord(flow_dim, self.context_net.context_dim, 1)
        elif flow_type == 'otflow':
            self.flow = OTFlow(flow_dim, 64, self.context_net.context_dim, 2)
        else:
            self.flow = build_nvp_flow(flow_dim, self.context_net.context_dim, flow_length)
        self.flow_type = flow_type
        self.register_buffer('prior_mu', torch.tensor(0.0))
        self.register_buffer('prior_scale', torch.tensor(1.0))

    def sample(self, start, goal, environment, N=1, z_environment=None, reconstruct=False):
        B, _ = start.shape
        prior = Normal(self.prior_mu, self.prior_scale)
        z = prior.sample(sample_shape=(B, N, self.H * self.du))
        return self.reconstruct(z, start, goal, environment, z_environment, reconstruct_env=reconstruct)

    def likelihood(self, u, start, goal, environment, z_environment=None, reconstruct=False):
        # assume that we may have multiple samples per environments, i.e. u is shape N x B x H x du
        B, N, H, du = u.shape
        assert H == self.H
        assert du == self.du
        context_dict = self.condition(start, goal, environment, z_environment, reconstruct=reconstruct)
        context = context_dict['context']
        context_n_samples = context.unsqueeze(1).repeat(1, N, 1).reshape(N * B, -1)
        log_qu = torch.zeros(B*N, device=u.device)

        out = self.flow(u.reshape(N * B, H * du), logpx=log_qu, context=context_n_samples, reverse=True)
        z = out[0]
        delta_log_qu = out[1]

        if self.training and self.flow_type == 'ffjord':
            context_dict['reg'] = self.flow.chain[0].regularization_states[1:]
        else:
            context_dict['reg'] = None

        prior = Normal(self.prior_mu, self.prior_scale)
        log_qu = prior.log_prob(z).sum(dim=1) - delta_log_qu
        return z, log_qu.reshape(B, N), context_dict

    def sample_w_peturbation(self, start, goal, environment, N=1, sigma=1, z_environment=None, reconstruct_env=False):
        B, _ = start.shape
        with torch.no_grad():
            u, _, context_dict = self.sample(start, goal, environment,
                                             z_environment=z_environment, N=N, reconstruct=False)
        # u = u.detach()
        # Note that we are sampling from a peturbed version of the qU, not qU itself.
        # We denote the peturbed distribution pU
        # We need prior weights which are qU / pU
        # pU is defined by p(U|U') Normal
        # and U' is distributed qU' with the flow

        # p_peturbation = Normal(loc=self.mean, scale=self.scale*sigma)
        peturbed_u = u + sigma * torch.randn_like(u)  # p_peturbation.sample(sample_shape=u.shape)
        z, log_qu, context_dict = self.likelihood(peturbed_u, start, goal, environment,
                                                  z_environment=z_environment,
                                                  reconstruct=reconstruct_env)

        if False:
            epsilon = peturbed_u.transpose(0, 1).reshape(B,
                                                         1, N, -1) - u.transpose(0, 1).reshape(B, N, 1, -1)
            # estimate pU via expectations
            log_pu = p_peturbation.log_prob(epsilon).sum(dim=-1)
            log_pu -= torch.max(log_pu)
            p_u = log_pu.exp().mean(dim=1).transpose(0, 1)
            p_u /= torch.sum(p_u, dim=0)
            q_u = (log_qu - torch.max(log_qu)).exp()
            q_u /= torch.sum(q_u, dim=0)
            sample_weights = q_u / (p_u + 1e-15)
            sample_weights /= torch.sum(sample_weights, dim=0) + 1e-6
            context_dict['Wj'] = sample_weights
        else:
            context_dict['Wj'] = None
        return peturbed_u, log_qu, context_dict

    def reconstruct(self, z, start, goal, environment, z_environment=None, reconstruct_env=False):
        B, N, dz = z.shape
        assert dz == self.H * self.du
        context_net_out = self.condition(start, goal, environment, z_environment, reconstruct=reconstruct_env)
        context = context_net_out['context']
        B, H = context.shape
        # Unfortunately we have to duplicate the context for each sample - there appears to be no way around this
        context_n_samples = context.unsqueeze(1).repeat(1, N, 1).reshape(B * N, H)
        prior = Normal(self.prior_mu, self.prior_scale)
        log_qz = torch.zeros(B*N, device=z.device)
        out = self.flow(z.reshape(N * B, -1), logpx=log_qz, context=context_n_samples, reverse=False)
        u, delta_log_qu = out[:2]
        if self.training and self.flow_type == 'ffjord':
            context_net_out['reg'] = self.flow.chain[0].regularization_states[1:]
        else:
            context_net_out['reg'] = None
        context_net_out['Wj'] = None
        log_qu = prior.log_prob(z.reshape(N * B, -1)).sum(dim=1) + delta_log_qu
        return u.reshape(B, N, self.H, self.du), log_qu.reshape(B, N), context_net_out

    def forward(self, start, goal, environment, N=1, sigma=None, reconstruct=False):
        if sigma is None:
            return self.sample(start, goal, environment, N, reconstruct=reconstruct)
        return self.sample_w_peturbation(start, goal, environment, N, sigma, reconstruct_env=reconstruct)
