import ipdb
import torch
from torch import nn
from flow_mpc import flows
from flow_mpc.encoders import Encoder, AttentionEncoder
from flow_mpc.flows.classifier import BinaryClassifier
from typing import List, Tuple


def build_realnvp_flow(flow_dim, context_dim, flow_length, hidden_dim, initialized=False):
    """
    Build a RealNVP flow consisting of multiple layers
    :param flow_dim: the dimension of variables to be flowed
    :param context_dim: the dimension of the conditioning variables
    :param flow_length: the number of the stacked flow layers
    :return: a RealNVP flow
    """
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
    return flows.SequentialFlow(flow_list)


def build_autoregressive_flow(state_dim, action_dim, channel_num, horizon, env_dim, flow_length, hidden_dim, initialized=False, extra_context_order=None, contact_dim=0):
    """
    Build an AutoRegressive flow consisting of multiple layers
    :param state_dim: the dimension of state variable
    :param action_dim: the dimension of action variable
    :param channel_num: the channel number of the variable to be flowed
    :param horizon: the number of future states to be predicted
    :param env_dim: the dimension of encoded environment information
    :param flow_length: the number of the stacked flow layers
    :param hidden_dim: the dimension of hidden layer in MLP
    :param initialized: indicate the initialization status of Actnorm layers
    :param contact_dim: the dimension of contact in one timestep
    :return: an AutoRegressive flow
    """
    flow_list = []
    extra_context_order = torch.zeros(0) if extra_context_order is None else extra_context_order
    forward_flag = True  # indicate whether the flow is doing forward inference or backward inference
    for _ in range(flow_length):
        if forward_flag:
            context_order = torch.cat((
                torch.zeros(state_dim),  # start state
                torch.arange(horizon).repeat(action_dim, 1).T.reshape(-1),  # action sequence
                torch.zeros(env_dim),  # environment code
                extra_context_order,  # extra context
                torch.zeros(1),  # noise magnitude
                torch.arange(horizon).repeat(contact_dim, 1).T.reshape(-1),  # contact sequence
            ))
        else:
            context_order = torch.cat((
                torch.zeros(state_dim),  # start state
                torch.arange(horizon).repeat(action_dim, 1).T.reshape(-1).__reversed__(),  # action sequence
                torch.zeros(env_dim),  # environment code
                extra_context_order,  # extra context
                torch.zeros(1),  # noise magnitude
                torch.arange(horizon).repeat(contact_dim, 1).T.reshape(-1).__reversed__(),  # contact sequence
            ))
        flow_list.append(flows.MaskedAutoRegressiveLayer(horizon=horizon, channel=channel_num,
                                                         context_order=context_order, intermediate_dim=hidden_dim))
        flow_list.append(flows.ActNorm(horizon*channel_num, initialized=initialized))
        flow_list.append(flows.Permutation(torch.arange(horizon*channel_num).__reversed__()))
        forward_flag = not forward_flag
    if flow_length > 0:
        if forward_flag:
            context_order = torch.cat((
                torch.zeros(state_dim),  # start state
                torch.arange(horizon).repeat(action_dim, 1).T.reshape(-1),  # action sequence
                torch.zeros(env_dim),  # environment code
                extra_context_order,  # extra context
                torch.zeros(1),  # noise magnitude
                torch.arange(horizon).repeat(contact_dim, 1).T.reshape(-1),  # contact sequence
            ))
        else:
            context_order = torch.cat((
                torch.zeros(state_dim),  # start state
                torch.arange(horizon).repeat(action_dim, 1).T.reshape(-1).__reversed__(),  # action sequence
                torch.zeros(env_dim),  # environment code
                extra_context_order,  # extra context
                torch.zeros(1),  # noise magnitude
                torch.arange(horizon).repeat(contact_dim, 1).T.reshape(-1).__reversed__(),  # contact sequence
            ))
        flow_list.append(flows.MaskedAutoRegressiveLayer(horizon=horizon, channel=channel_num,
                                                         context_order=context_order, intermediate_dim=hidden_dim))
    return flows.SequentialFlow(flow_list)


def build_multi_scale_autoregressive_flow(state_dim, action_dim, channel_num, horizon, env_dim, hidden_dim,
                                          initialized=False, extra_context_order=None, condition_prior=True):
    """
    Build an AutoRegressive flow consisting of multiple layers with different scale depending on horizon
    :param state_dim: the dimension of state variable
    :param action_dim: the dimension of action variable
    :param channel_num: the channel number of the variable to be flowed
    :param horizon: the number of future states to be predicted
    :param env_dim: the dimension of encoded environment information
    :param hidden_dim: the dimension of hidden layer in MLP
    :param initialized: indicate the initialization status of Actnorm layers
    :param with_contact: indicate whether contact flag is used
    :return: an AutoRegressive flow
    """
    flow_list = []
    extra_context_order = torch.zeros(0) if extra_context_order is None else extra_context_order
    forward_flag = False  # indicate whether the flow is doing forward inference or backward inference
    current_horizon = 1
    flow_length = 2*horizon
    for i in range(flow_length):
        if forward_flag:
            context_order = torch.cat((
                torch.zeros(state_dim),  # start state
                torch.arange(horizon).repeat(action_dim, 1).T.reshape(-1),  # action sequence
                torch.zeros(env_dim),  # environment code
                extra_context_order,  # extra context
                torch.zeros(1),  # noise magnitude
            ))
        else:
            context_order = torch.cat((
                torch.zeros(state_dim),  # start state
                torch.arange(horizon).repeat(action_dim, 1).T.reshape(-1).__reversed__(),  # action sequence
                torch.zeros(env_dim),  # environment code
                extra_context_order,  # extra context
                torch.zeros(1),  # noise magnitude
            ))
        if i != 0 and not forward_flag:
            # mask out the future action input
            if condition_prior:
                context_mask = (context_order <= current_horizon)
                flow_list.append(flows.ConditionalSplitFlow(z_dim=current_horizon*channel_num + channel_num,
                                                            z_split_dim=channel_num,
                                                            context_dim=int(context_mask.sum()), hidden_dim=hidden_dim,
                                                            context_mask=context_mask))
            else:
                # flow_list.append(flows.SplitFlow(z_split_dim=channel_num))
                context_mask = torch.zeros_like(context_order)
                flow_list.append(flows.ConditionalSplitFlow(z_dim=current_horizon * channel_num + channel_num,
                                                            z_split_dim=channel_num,
                                                            context_dim=int(context_mask.sum()), hidden_dim=hidden_dim,
                                                            context_mask=context_mask))
            current_horizon += 1
        flow_list.append(flows.MaskedAutoRegressiveLayer(horizon=current_horizon, channel=channel_num,
                                                         context_order=context_order, intermediate_dim=hidden_dim))
        flow_list.append(flows.ActNorm(current_horizon*channel_num, initialized=initialized))
        flow_list.append(flows.Permutation(torch.arange(current_horizon*channel_num).__reversed__()))
        forward_flag = not forward_flag
    if flow_length > 0:
        assert current_horizon == horizon
        if forward_flag:
            context_order = torch.cat((
                torch.zeros(state_dim),  # start state
                torch.arange(horizon).repeat(action_dim, 1).T.reshape(-1),  # action sequence
                torch.zeros(env_dim),  # environment code
                extra_context_order,  # extra context
                torch.zeros(1),  # noise magnitude
            ))
        else:
            context_order = torch.cat((
                torch.zeros(state_dim),  # start state
                torch.arange(horizon).repeat(action_dim, 1).T.reshape(-1).__reversed__(),  # action sequence
                torch.zeros(env_dim),  # environment code
                extra_context_order,  # extra context
                torch.zeros(1),  # noise magnitude
            ))
        flow_list.append(flows.MaskedAutoRegressiveLayer(horizon=horizon, channel=channel_num,
                                                         context_order=context_order, intermediate_dim=hidden_dim))
        flow_list.append(flows.Permutation(torch.arange(horizon * channel_num).__reversed__()))
    return flows.SequentialFlow(flow_list)


def build_coupling_autoregressive_flow(state_dim, action_dim, horizon, env_dim, flow_length, hidden_dim, initialized=False):
    """
    Build a coupling autoregressive flow consisting of two autoregressive flows
    :param state_dim: the dimension of state variable
    :param action_dim: the dimension of action variable
    :param horizon: the number of future states to be predicted
    :param env_dim: the dimension of encoded environment information
    :param flow_length: the number of the stacked flow layers
    :param hidden_dim: the dimension of hidden layer in MLP
    :param initialized: indicate the initialization status of Actnorm layers
    :return: a coupling autoregressive flow
    """
    contact_context_order = torch.cat((
            torch.zeros(state_dim),  # start state
            torch.arange(horizon).repeat(action_dim, 1).T.reshape(-1),  # action sequence
            torch.zeros(env_dim),  # environment code
            torch.zeros(1),  # noise magnitude
            torch.arange(horizon).repeat(state_dim, 1).T.reshape(-1),   # state sequence
        ))
    dynamic_context_order = torch.cat((
        torch.zeros(state_dim),  # start state
        torch.arange(horizon).repeat(action_dim, 1).T.reshape(-1),  # action sequence
        torch.zeros(env_dim),  # environment code
        torch.zeros(1),  # noise magnitude
        torch.arange(horizon),  # contact sequence
    ))

    contact_flow_list = []
    dynamic_flow_list = []
    for _ in range(flow_length):
        contact_flow_list.append(flows.MaskedAutoRegressiveLayer(horizon=horizon, channel=1, context_order=contact_context_order, intermediate_dim=hidden_dim))
        contact_flow_list.append(flows.ActNorm(horizon, initialized=initialized))
        contact_flow_list.append(flows.Permutation(torch.arange(horizon).__reversed__()))

        dynamic_flow_list.append(flows.MaskedAutoRegressiveLayer(horizon=horizon, channel=state_dim, context_order=dynamic_context_order, intermediate_dim=hidden_dim))
        dynamic_flow_list.append(flows.ActNorm(horizon*state_dim, initialized=initialized))
        dynamic_flow_list.append(flows.Permutation(torch.arange(horizon*state_dim).__reversed__()))
    if flow_length > 0:
        contact_flow_list.append(flows.MaskedAutoRegressiveLayer(horizon=horizon, channel=1, context_order=contact_context_order, intermediate_dim=hidden_dim))
        dynamic_flow_list.append(flows.MaskedAutoRegressiveLayer(horizon=horizon, channel=state_dim, context_order=dynamic_context_order, intermediate_dim=hidden_dim))

    return flows.CouplingSequentialFlow(contact_flow_list, dynamic_flow_list)


class FlowModel(nn.Module):
    """The dynamic model based on normalizing flow"""
    def __init__(self, state_dim, action_dim, horizon, hidden_dim=256, flow_length=10, condition=True,
                 initialized=False, flow_type='nvp',
                 state_mean=None, state_std=None,
                 action_mean=None, action_std=None,
                 ):
        super(FlowModel, self).__init__()
        self.flow_type = flow_type
        if flow_type == 'nvp':
            self.flow = build_realnvp_flow(flow_dim=state_dim * horizon,
                                           context_dim=state_dim + action_dim * horizon,
                                           flow_length=flow_length, hidden_dim=hidden_dim, initialized=initialized)
        elif flow_type == 'autoregressive':
            self.flow = build_autoregressive_flow(state_dim=state_dim, action_dim=action_dim, channel_num=state_dim,
                                                  horizon=horizon, env_dim=0, flow_length=flow_length,
                                                  hidden_dim=hidden_dim, initialized=initialized, extra_context_order=None)
        else:
            raise NotImplementedError
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon

        self.register_buffer('state_mean', torch.tensor(state_mean, dtype=torch.float) if state_mean is not None else torch.zeros(state_dim))
        self.register_buffer('state_std', torch.tensor(state_std, dtype=torch.float) if state_std is not None else torch.ones(state_dim))
        self.register_buffer('action_mean', torch.tensor(action_mean, dtype=torch.float) if action_mean is not None else torch.zeros(action_dim))
        self.register_buffer('action_std', torch.tensor(action_std, dtype=torch.float) if action_std is not None else torch.ones(action_dim))

        if condition:
            self.prior = flows.ConditionalPrior(state_dim + action_dim * horizon + 1, state_dim * horizon, hidden_dim=hidden_dim)
        else:
            self.prior = flows.GaussianPrior(state_dim * horizon)

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
        :return log_prob: the probability of generating the predicted trajectory of shape (B,)
        """
        batch_size = start_state.shape[0]
        start_state = (start_state - self.state_mean) / self.state_std
        action = (action - self.action_mean) / self.action_std
        context = torch.cat((start_state, action.reshape(batch_size, -1)), dim=1)
        if not reverse: # sampling
            noise_magnitude = context.new_zeros(batch_size, 1)
            context = torch.cat((context, noise_magnitude), dim=1)
            z, log_prob = self.prior(z=None, logpx=0, context=context, reverse=reverse)
            x, ldj = self.flow(z, logpx=0, context=context, reverse=reverse)
            # x, ldj = z, 0
            relative_displacement = x.reshape(batch_size, -1, self.state_dim)
            traj = start_state.unsqueeze(1) + torch.cumsum(relative_displacement, dim=1)
            traj = traj * self.state_std + self.state_mean
            return {"traj": traj, "logp": log_prob + ldj}
        else:           # training
            noise_magnitude = torch.rand((batch_size, 1), dtype=context.dtype, device=context.device) * 2 - 1  # (-1, 1)
            context = torch.cat((context, noise_magnitude), dim=1)
            traj = (traj - self.state_mean) / self.state_std
            traj_noise = torch.randn(traj.shape, dtype=traj.dtype, device=traj.device) * noise_magnitude.abs().unsqueeze(-1)
            traj = traj + traj_noise
            before_traj = torch.cat((start_state.unsqueeze(1), traj[:, :-1]), dim=1)
            relative_displacement = traj - before_traj
            x = relative_displacement.reshape(batch_size, -1)
            z, ldj = self.flow(x, logpx=0, context=context, reverse=reverse)
            # z, ldj = x, 0
            z, log_prob = self.prior(z=z, logpx=ldj, context=context, reverse=reverse)
            return {"z": z, "logp": log_prob}

class ImageFlowModel(nn.Module):
    """The dynamics model based on normalizing flow, along with an extra encoder to encode environment image"""
    def __init__(self, state_dim, action_dim, horizon, image_size: Tuple[int, int],
                 env_dim=64, hidden_dim=256, flow_length=10, condition=True, initialized=False,
                 flow_type='autoregressive', with_contact=False, relative_displacement=True,
                 contact_dim=0, pre_rotation=False, prior_pretrain=False, aligner=None,
                 state_mean=None, state_std=None,
                 action_mean=None, action_std=None,
                 image_mean=None, image_std=None,
                 flow_mean=None, flow_std=None,
                 ):
        """
        Explanation about some flag parameters
        :param condition: whether to enable conditional prior
        :param initialized: Actnorm layers initialization status
        :param with_contact: whether to take contact flags as model input (deprecated)
        :param relative_displacement: whether to use relative displacement as flow i/o (model i/o not influenced)
        :param contact_dim: the contact flag dimension at one timestep. 0 means binary classification in latent space not enabled
        :param pre_rotation: whether to rotation the context variables in prior
        """
        super(ImageFlowModel, self).__init__()
        self.flow_type = flow_type
        # self.contact_input = with_contact
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.image_size = image_size
        self.env_dim = env_dim
        self.relative_displacement = relative_displacement
        self.pre_rotation = pre_rotation
        self.aligner = aligner
        self.prior_pretrain = prior_pretrain
        self.register_buffer('state_mean', torch.tensor(state_mean, dtype=torch.float) if state_mean is not None else torch.zeros(state_dim))
        self.register_buffer('state_std', torch.tensor(state_std, dtype=torch.float) if state_std is not None else torch.ones(state_dim))
        self.register_buffer('action_mean', torch.tensor(action_mean, dtype=torch.float) if action_mean is not None else torch.zeros(action_dim))
        self.register_buffer('action_std', torch.tensor(action_std, dtype=torch.float) if action_std is not None else torch.ones(action_dim))
        self.register_buffer('image_mean', torch.tensor(image_mean, dtype=torch.float) if image_mean is not None else torch.zeros(()))
        self.register_buffer('image_std', torch.tensor(image_std, dtype=torch.float) if image_std is not None else torch.ones(()))
        self.register_buffer('flow_mean', torch.tensor(flow_mean, dtype=torch.float) if flow_mean is not None else torch.zeros((state_dim)))
        self.register_buffer('flow_std', torch.tensor(flow_std, dtype=torch.float) if flow_std is not None else torch.ones((state_dim)))

        extra_context_order = torch.arange(horizon).repeat(contact_dim, 1).T.reshape(-1) if contact_dim else None
        if flow_type == 'nvp':
            self.flow = build_realnvp_flow(flow_dim=state_dim * horizon,
                                           context_dim=state_dim + action_dim * horizon + env_dim + 1,
                                           flow_length=flow_length, hidden_dim=hidden_dim, initialized=initialized)
        elif flow_type == 'autoregressive':
            extra_context_order = None
            self.flow = build_autoregressive_flow(state_dim=state_dim, action_dim=action_dim, channel_num=state_dim,
                                                  horizon=horizon, env_dim=env_dim, flow_length=flow_length, contact_dim=contact_dim,
                                                  hidden_dim=hidden_dim, initialized=initialized, extra_context_order=extra_context_order)
        elif flow_type == 'msar':
            extra_context_order = None
            self.flow = build_multi_scale_autoregressive_flow(state_dim=state_dim, action_dim=action_dim, channel_num=state_dim,
                                                              horizon=horizon, env_dim=env_dim, hidden_dim=hidden_dim,
                                                              initialized=initialized, extra_context_order=extra_context_order,
                                                              condition_prior=condition)
        else:
            raise NotImplementedError(f"flow type {flow_type} not recognizable.")
        initial_horizon = 1 if flow_type == 'msar' else horizon
        if self.prior_pretrain:
            self.prior = flows.ConditionalPrior(state_dim + action_dim * horizon + 1,
                                                state_dim * initial_horizon, hidden_dim=hidden_dim)
        else:
            if condition:
                context_order = torch.cat((
                    torch.zeros(state_dim),  # start state
                    torch.arange(horizon).repeat(action_dim, 1).T.reshape(-1),  # action sequence
                    torch.zeros(env_dim),  # environment code
                    torch.zeros(1)  # noise magnitude
                ))
                self.prior = flows.ConditionalPrior(state_dim + action_dim * horizon + env_dim + 1,
                                                    state_dim * initial_horizon, hidden_dim=hidden_dim,
                                                    context_order=context_order)
            else:
                self.prior = flows.GaussianPrior(state_dim * initial_horizon)

        self.encoder = Encoder(image_size, env_dim)
        # self.s_u_encoder = nn.Sequential(nn.Linear(state_dim + action_dim * horizon, hidden_dim),
        #                                  nn.ReLU(),
        #                                  nn.Linear(hidden_dim, env_dim)
        #                                  )
        # self.s_u_encoder.requires_grad_(False)
        if contact_dim > 0:
            self.latent_classifier = BinaryClassifier(state_dim*initial_horizon + state_dim + action_dim*horizon + env_dim, contact_dim*horizon, hidden_dim)
            print("latent classifier enabled!")

    def forward(self, start_state: torch.Tensor, action: torch.Tensor, image: torch.Tensor, reconstruct=False, reverse=False, z=None, traj=None, contact_flag=None):
        """
        Forward:
        Given state at start time, actions along a time horizon, and an environment image, sample a predicted state trajectory and return its log prob
        Reverse:
        Given state at start time, actions along a time horizon, an environment image and a state trajectory, return the corresponding latent variable and its log_prob
        :param start_state: torch tensor of shape (B, state_dim)
        :param action: torch tensor of shape (B, horizon, action_dim)
        :param image: torch tensor of shape (B/N, channel, height, width)
        :param reconstruct: whether or not to reconstruct the original image
        :param reverse: False means sampling and using this model, True means using passed-in trajectory to get the latent variable and its prob
        :param z: given latent variable instead of sampling from prior
        :param traj: torch tensor of shape (B, horizon, channel_num)
        :param contact_flag: torch tensor of shape (B, contact_dim*horizon) OR None
        :return predicted_traj of shape (B, horizon, channel_num) OR latent variable of shape (B, horizon x channel_num) depending on reverse or not
        :return log probability of sampling the trajectory of shape (B,)
        :return reconstructed image that has the same shape as input image
        :return predicted contact flag scores of shape (B, horizon*contact_dim, 2)
        """
        start_state = (start_state - self.state_mean) / self.state_std
        action = (action - self.action_mean) / self.action_std
        image = (image - self.image_mean) / self.image_std
        angle = None
        if self.pre_rotation:
            angle = -torch.atan2(start_state[:, 3], start_state[:, 2])
            state_forward_rotation = self.get_rotation_matrix(angle, self.state_dim)
            action_forward_rotation = self.get_rotation_matrix(angle, self.action_dim)
            state_backward_rotation = self.get_rotation_matrix(-angle, self.state_dim)
            start_state = (state_forward_rotation @ start_state.unsqueeze(-1)).squeeze(-1)
            action = (action_forward_rotation.unsqueeze(1) @ action.unsqueeze(-1)).squeeze(-1)
        batch_size = start_state.shape[0]
        env_code = self.encoder.encode(image, angle=angle)   # shape of (B/N, env_dim)
        if reconstruct:
            image_reconstruct = self.encoder.reconstruct(env_code) * self.image_std + self.image_mean
        else:
            image_reconstruct = None
        N = start_state.shape[0] // env_code.shape[0]
        env_code = env_code.unsqueeze(0).repeat(N, 1, 1).transpose(0, 1).reshape(-1, env_code.shape[1])
        basic_context = torch.cat((start_state, action.reshape(batch_size, -1), env_code), dim=1)
        if not reverse:     # sampling
            noise_magnitude = basic_context.new_zeros(batch_size, 1)
            context = torch.cat((basic_context, noise_magnitude), dim=1)
            prior_context = context if not self.prior_pretrain else torch.cat((start_state, action.reshape(batch_size, -1), noise_magnitude), dim=1)
            z, log_prob = self.prior(z=z, logpx=0, context=prior_context, reverse=reverse)
            contact_prediction_score = None
            if hasattr(self, "latent_classifier"):
                contact_prediction_score = self.latent_classifier(torch.cat((z, basic_context), dim=1))
                contact_prediction = (contact_prediction_score > 0).float() * 2 - 1
                context = torch.cat((context, contact_prediction), dim=1)
            x, ldj = self.flow(z, logpx=0, context=context, reverse=reverse)
            if self.relative_displacement:
                relative_displacement = x.reshape(batch_size, -1, self.state_dim)
                traj = start_state.unsqueeze(1) + torch.cumsum(relative_displacement, dim=1)
            else:
                traj = x.reshape(batch_size, -1, self.state_num)
            if self.pre_rotation:
                traj = (state_backward_rotation.unsqueeze(1) @ traj.unsqueeze(-1)).squeeze(-1)
            traj = traj * self.flow_std + self.flow_mean
            alignment = self.aligner(image, traj, cross_align=False) if self.aligner is not None else None
            return {"traj": traj, "logp": log_prob + ldj, "image_reconstruct": image_reconstruct, "contact_logit": contact_prediction_score, "alignment": alignment}
        else:           # training
            assert traj is not None
            noise_magnitude = torch.rand((batch_size, 1), dtype=basic_context.dtype, device=basic_context.device) * 2 - 1   # (-1, 1)
            context = torch.cat((basic_context, noise_magnitude), dim=1)
            if hasattr(self, "latent_classifier"):
                assert contact_flag is not None
                contact_flag = contact_flag*2-1
                context = torch.cat((context, contact_flag), dim=1)
            traj = (traj - self.flow_mean) / self.flow_std
            if self.pre_rotation:
                traj = (state_forward_rotation.unsqueeze(1) @ traj.unsqueeze(-1)).squeeze(-1)
            traj_noise = torch.randn(traj.shape, dtype=traj.dtype, device=traj.device) * noise_magnitude.abs().unsqueeze(-1)
            traj = traj + traj_noise
            if self.relative_displacement:
                before_traj = torch.cat((start_state.unsqueeze(1), traj[:, :-1]), dim=1)
                relative_displacement = traj - before_traj
                x = relative_displacement.reshape(batch_size, -1)
            else:
                x = traj.reshape(batch_size, -1)
            z, ldj = self.flow(x, logpx=0, context=context, reverse=reverse)
            # z, ldj = x, 0
            prior_context = torch.cat((basic_context, noise_magnitude), dim=1) if not self.prior_pretrain \
                else torch.cat((start_state, action.reshape(batch_size, -1), noise_magnitude), dim=1)
            z, log_prob = self.prior(z=z, logpx=ldj, context=prior_context, reverse=reverse)
            contact_prediction_score = self.latent_classifier(torch.cat((z, basic_context), dim=1)) if hasattr(self, "latent_classifier") else None
            return {"z": z, "logp": log_prob, "image_reconstruct": image_reconstruct, "contact_logit": contact_prediction_score}

    def get_rotation_matrix(self, angle: torch.Tensor, state_num: int = None):
        """
        Get the 2d rotation matrix applied to the state variables
        :param angle: a tensor of shape (B,) giving a batch of rotation angles in radian
        :param state_num: number of states to be rotated, must be even in 2d case.
        :return rotation_matrix: a rotation matrix of shape (B, state_num, state_num)
        """
        # angle = angle / 180 * torch.pi
        if state_num is None:
            state_num = self.state_dim
        assert state_num % 2 == 0
        rotation_matrix = angle.new_zeros(angle.shape[0], state_num, state_num)
        x_vector = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1).repeat(1, state_num // 2)
        y_vector = torch.stack([-torch.sin(angle), torch.cos(angle)], dim=1).repeat(1, state_num // 2)
        x_idx = torch.arange(0, state_num, 2).repeat(2, 1).T.flatten()
        y_idx = torch.arange(1, state_num, 2).repeat(2, 1).T.flatten()
        rotation_matrix[:, torch.arange(state_num), x_idx] = x_vector
        rotation_matrix[:, torch.arange(state_num), y_idx] = y_vector

        return rotation_matrix


class DoubleImageFlowModel(nn.Module):
    """The dynamic model that has a hierarchical structure to predict contact and dynamics separately"""
    def __init__(self, state_dim, action_dim, horizon, image_size: Tuple[int, int],
                 env_dim=64, hidden_dim=256, flow_length=10, condition=True, initialized=False,
                 flow_type='autoregressive',
                 state_mean=None, state_std=None,
                 action_mean=None, action_std=None,
                 image_mean=None, image_std=None,
                 ):
        super(DoubleImageFlowModel, self).__init__()
        self.flow_type = flow_type
        if flow_type == 'nvp':
            self.contact_flow = build_realnvp_flow(flow_dim=horizon,
                                           context_dim=state_dim + action_dim * horizon + env_dim + 1,
                                           flow_length=flow_length, hidden_dim=hidden_dim, initialized=initialized)
            self.dynamic_flow = build_realnvp_flow(flow_dim=state_dim * horizon,
                                           context_dim=state_dim + action_dim * horizon + env_dim + 1,
                                           flow_length=flow_length, hidden_dim=hidden_dim, initialized=initialized)
        elif flow_type == 'autoregressive':
            self.contact_flow = build_autoregressive_flow(state_dim=state_dim, action_dim=action_dim, channel_num=1,
                                                          horizon=horizon, env_dim=env_dim, flow_length=flow_length,
                                                          hidden_dim=hidden_dim, initialized=initialized,
                                                          # extra_context_order=torch.arange(horizon).repeat(state_dim, 1).T.reshape(-1))
                                                          extra_context_order=None)
            self.dynamic_flow = build_autoregressive_flow(state_dim=state_dim, action_dim=action_dim, channel_num=state_dim,
                                                          horizon=horizon, env_dim=env_dim, flow_length=flow_length,
                                                          hidden_dim=hidden_dim, initialized=initialized,
                                                          extra_context_order=torch.arange(horizon))
        else:
            raise NotImplementedError(f"flow type {flow_type} not recognizable.")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.image_size = image_size
        self.env_dim = env_dim
        self.register_buffer('state_mean', torch.tensor(state_mean, dtype=torch.float) if state_mean is not None else torch.zeros(state_dim))
        self.register_buffer('state_std', torch.tensor(state_std, dtype=torch.float) if state_std is not None else torch.ones(state_dim))
        self.register_buffer('action_mean', torch.tensor(action_mean, dtype=torch.float) if action_mean is not None else torch.zeros(action_dim))
        self.register_buffer('action_std', torch.tensor(action_std, dtype=torch.float) if action_std is not None else torch.ones(action_dim))
        self.register_buffer('image_mean', torch.tensor(image_mean, dtype=torch.float) if image_mean is not None else torch.zeros(()))
        self.register_buffer('image_std', torch.tensor(image_std, dtype=torch.float) if image_std is not None else torch.ones(()))

        if condition:
            context_order = torch.cat((
                torch.zeros(state_dim),  # start state
                torch.arange(horizon).repeat(action_dim, 1).T.reshape(-1),  # action sequence
                torch.zeros(env_dim),  # environment code
                # torch.arange(horizon).repeat(state_dim, 1).T.reshape(-1),   # future state
                torch.zeros(1)  # noise magnitude
            ))
            self.contact_prior = flows.ConditionalPrior(state_dim + action_dim*horizon + env_dim + 1,
                                                        horizon, hidden_dim=hidden_dim,
                                                        context_order=context_order)
            context_order = torch.cat((
                torch.zeros(state_dim),  # start state
                torch.arange(horizon).repeat(action_dim, 1).T.reshape(-1),  # action sequence
                torch.zeros(env_dim),  # environment code
                torch.arange(horizon),  # contact flag
                torch.zeros(1)  # noise magnitude
            ))
            self.dynamic_prior = flows.ConditionalPrior(state_dim + action_dim*horizon + env_dim + horizon + 1,
                                                        state_dim * horizon, hidden_dim=hidden_dim,
                                                        context_order=context_order)
        else:
            self.contact_prior = flows.GaussianPrior(horizon)
            self.dynamic_prior = flows.GaussianPrior(state_dim * horizon)

        self.encoder = Encoder(image_size, env_dim)
        # self.encoder = AttentionEncoder(context_dim=state_dim + action_dim*horizon + 1, z_env_dim=env_dim)

    def forward(self, start_state: torch.Tensor, action: torch.Tensor, image: torch.Tensor, reconstruct=False, reverse=False, traj=None, contact_flag=None, output_contact=False):
        """
        Forward:
        Given state at start time, actions along a time horizon, and an environment image, sample a predicted state trajectory and return its log prob
        Reverse:
        Given state at start time, actions along a time horizon, an environment image and a state trajectory, return the corresponding latent variable and its log_prob
        :param start_state: torch tensor of shape (B, state_dim)
        :param action: torch tensor of shape (B, horizon, action_dim)
        :param image: torch tensor of shape (B/N, channel, height, width)
        :param reverse: False means sampling and using this model, True means using passed-in trajectory to get the latent variable and its prob
        :param traj: torch tensor of shape (B, horizon, state_dim)
        :param contact_flag: torch tensor of shape (B, horizon) OR None
        :param output_contact: whether or not to output the contact variable
        :return: predicted_traj of shape (B, horizon, state_dim) OR latent variable of shape (B, horizon x state_dim) depending on reverse or not
        :return: log probability of sampling the trajectory of shape (B,)
        """
        start_state = (start_state - self.state_mean) / self.state_std
        action = (action - self.action_mean) / self.action_std
        image = (image - self.image_mean) / self.image_std
        batch_size = start_state.shape[0]
        env_code = self.encoder.encode(image)  # shape of (B/N, env_dim)
        if reconstruct:
            image_reconstruct = self.encoder.reconstruct(env_code) * self.image_std + self.image_mean
        else:
            image_reconstruct = None
        N = start_state.shape[0] // env_code.shape[0]
        env_code = env_code.unsqueeze(0).repeat(N, 1, 1).transpose(0, 1).reshape(-1, env_code.shape[1])
        if not reverse:  # sampling
            noise_magnitude = start_state.new_zeros(batch_size, 1)
            # encoder_context = torch.cat((start_state, action.reshape(batch_size, -1), noise_magnitude), dim=1)
            # env_code, _ = self.encoder.encode(image, encoder_context)
            # x_contact = start_state.new_zeros(batch_size, self.horizon)
            for _ in range(1):
                # contact_context = torch.cat((start_state, action.reshape(batch_size, -1), env_code, x_dynamic, noise_magnitude), dim=1)
                contact_context = torch.cat((start_state, action.reshape(batch_size, -1), env_code, noise_magnitude), dim=1)
                z_contact, log_prob = self.contact_prior(z=None, logpx=0, context=contact_context, reverse=reverse)
                x_contact, log_prob = self.contact_flow(z_contact, logpx=log_prob, context=contact_context, reverse=reverse)
                x_contact = (x_contact > 0).float() * 2 - 1

                dynamic_context = torch.cat((start_state, action.reshape(batch_size, -1), env_code, x_contact, noise_magnitude), dim=1)
                z_dynamic, log_prob = self.dynamic_prior(z=None, logpx=log_prob, context=dynamic_context, reverse=reverse)
                x_dynamic, log_prob = self.dynamic_flow(z_dynamic, logpx=log_prob, context=dynamic_context, reverse=reverse)
                x_dynamic = x_dynamic.reshape(batch_size, -1, self.state_dim)
                x_dynamic = (start_state.unsqueeze(1) + torch.cumsum(x_dynamic, dim=1)).reshape(batch_size, -1)


            # contact_flag = (contact_flag.squeeze(-1) > 0).float()
            # dynamic_context = torch.cat((start_state, action.reshape(batch_size, -1), env_code, contact_flag, noise_magnitude), dim=1)
            # z, log_prob = self.dynamic_prior(z=None, logpx=log_prob, context=dynamic_context, reverse=reverse)
            # x, log_prob = self.dynamic_flow(z, logpx=log_prob, context=dynamic_context, reverse=reverse)
            traj = x_dynamic.reshape(batch_size, -1, self.state_dim)
            traj = traj * self.state_std + self.state_mean
            if not output_contact:
                return traj, log_prob, image_reconstruct
            else:
                contact_flag = (x_contact + 1) / 2
                return traj, contact_flag, log_prob, image_reconstruct
        else:           # training
            assert traj is not None
            assert contact_flag is not None
            traj = (traj - self.state_mean) / self.state_std
            contact_flag = (contact_flag - 0.5) / 0.5
            noise_magnitude = torch.rand((batch_size, 1), dtype=action.dtype, device=action.device) * 2 - 1  # (-1, 1)
            # encoder_context = torch.cat((start_state, action.reshape(batch_size, -1), noise_magnitude), dim=1)
            # env_code, _ = self.encoder.encode(image, encoder_context)
            # dynamic_noise_magnitude = torch.rand((batch_size, 1), dtype=action.dtype, device=action.device) * 2 - 1  # (-1, 1)
            # contact_noise_magnitude = torch.rand((batch_size, 1), dtype=action.dtype, device=action.device) * 2 - 1  # (-1, 1)
            dynamic_context = torch.cat((start_state, action.reshape(batch_size, -1), env_code, contact_flag, noise_magnitude), dim=1)
            # contact_context = torch.cat((start_state, action.reshape(batch_size, -1), env_code, traj.reshape(batch_size, -1), contact_noise_magnitude), dim=1)
            contact_context = torch.cat((start_state, action.reshape(batch_size, -1), env_code, noise_magnitude), dim=1)

            traj_noise = torch.randn(traj.shape, dtype=traj.dtype, device=traj.device) * noise_magnitude.abs().unsqueeze(-1)
            traj = traj + traj_noise
            contact_noise = torch.randn(contact_flag.shape, dtype=contact_flag.dtype, device=contact_flag.device) * noise_magnitude.abs()
            contact_flag = contact_flag + contact_noise

            before_traj = torch.cat((start_state.unsqueeze(1), traj[:, :-1]), dim=1)
            relative_traj = traj - before_traj
            relative_traj = relative_traj.reshape(batch_size, -1)

            z_contact, log_prob_contact = self.contact_flow(contact_flag, logpx=0, context=contact_context, reverse=reverse)
            z_contact, log_prob_contact = self.contact_prior(z=z_contact, logpx=log_prob_contact, context=contact_context, reverse=reverse)
            z_dynamic, log_prob_dynamic = self.dynamic_flow(relative_traj, logpx=0, context=dynamic_context, reverse=reverse)
            z_dynamic, log_prob_dynamic = self.dynamic_prior(z=z_dynamic, logpx=log_prob_dynamic, context=dynamic_context, reverse=reverse)
            if not output_contact:
                return z_dynamic, log_prob_contact + log_prob_dynamic, image_reconstruct
            else:
                return z_dynamic, z_contact, log_prob_contact + log_prob_dynamic, image_reconstruct


class CouplingImageFlowModel(nn.Module):
    """The dynamic model that has a coupling hierarchical structure to predict contact and dynamics separately"""
    def __init__(self, state_dim, action_dim, horizon, image_size: Tuple[int, int],
                 env_dim=64, hidden_dim=256, flow_length=10, condition=True, initialized=False,
                 flow_type='autoregressive',
                 state_mean=None, state_std=None,
                 action_mean=None, action_std=None,
                 image_mean=None, image_std=None,
                 ):
        super(CouplingImageFlowModel, self).__init__()
        self.flow_type = flow_type
        if flow_type == 'autoregressive':
            self.flow = build_coupling_autoregressive_flow(state_dim=state_dim, action_dim=action_dim, horizon=horizon,
                                                           env_dim=env_dim, flow_length=flow_length,
                                                           hidden_dim=hidden_dim, initialized=initialized)
        else:
            raise NotImplementedError(f"flow type {flow_type} not implemented.")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.image_size = image_size
        self.env_dim = env_dim
        self.register_buffer('state_mean', torch.tensor(state_mean, dtype=torch.float) if state_mean is not None else torch.zeros(state_dim))
        self.register_buffer('state_std', torch.tensor(state_std, dtype=torch.float) if state_std is not None else torch.ones(state_dim))
        self.register_buffer('action_mean', torch.tensor(action_mean, dtype=torch.float) if action_mean is not None else torch.zeros(action_dim))
        self.register_buffer('action_std', torch.tensor(action_std, dtype=torch.float) if action_std is not None else torch.ones(action_dim))
        self.register_buffer('image_mean', torch.tensor(image_mean, dtype=torch.float) if image_mean is not None else torch.zeros(()))
        self.register_buffer('image_std', torch.tensor(image_std, dtype=torch.float) if image_std is not None else torch.ones(()))

        if condition:
            context_order = torch.cat((
                torch.zeros(state_dim),  # start state
                torch.arange(horizon).repeat(action_dim, 1).T.reshape(-1),  # action sequence
                torch.zeros(env_dim),  # environment code
                torch.zeros(1)  # noise magnitude
            ))
            self.contact_prior = flows.ConditionalPrior(state_dim + action_dim * horizon + env_dim + 1,
                                                        horizon, hidden_dim=hidden_dim,
                                                        context_order=context_order)
            self.dynamic_prior = flows.ConditionalPrior(state_dim + action_dim * horizon + env_dim + 1,
                                                        state_dim * horizon, hidden_dim=hidden_dim,
                                                        context_order=context_order)
        else:
            self.contact_prior = flows.GaussianPrior(horizon)
            self.dynamic_prior = flows.GaussianPrior(state_dim * horizon)

        self.encoder = AttentionEncoder(context_dim=state_dim + action_dim * horizon + 1, z_env_dim=64)
        # self.encoder = Encoder(image_size, env_dim)

    def forward(self, start_state: torch.Tensor, action: torch.Tensor, image: torch.Tensor, reverse=False, traj=None, contact_flag=None, output_contact=False):
        """
        Forward:
        Given state at start time, actions along a time horizon, and an environment image, sample a predicted state trajectory and return its log prob
        Reverse:
        Given state at start time, actions along a time horizon, an environment image and a state trajectory, return the corresponding latent variable and its log_prob
        :param start_state: torch tensor of shape (B, state_dim)
        :param action: torch tensor of shape (B, horizon, action_dim)
        :param image: torch tensor of shape (B/N, channel, height, width)
        :param reverse: False means sampling and using this model, True means using passed-in trajectory to get the latent variable and its prob
        :param traj: torch tensor of shape (B, horizon, state_dim)
        :param contact_flag: torch tensor of shape (B, horizon) OR None
        :param output_contact: whether or not to output the contact variable
        :return: predicted trajectory of shape (B, horizon, state_dim) OR latent variable of shape (B, horizon x state_dim) depending on reverse or not
        :return: (optional) predicted contact flags of shape (B, horizon) OR latent variable of shape (B, horizon) depending on reverse or not
        :return: log probability of sampling the trajectory of shape (B,)
        """
        start_state = (start_state - self.state_mean) / self.state_std
        action = (action - self.action_mean) / self.action_std
        image = (image - self.image_mean) / self.image_std
        batch_size = start_state.shape[0]
        # env_code = self.encoder.encode(image)  # shape of (B/N, env_dim)
        # N = start_state.shape[0] // env_code.shape[0]
        # env_code = env_code.unsqueeze(0).repeat(N, 1, 1).transpose(0, 1).reshape(-1, env_code.shape[1])
        if not reverse:  # sampling
            noise_magnitude = start_state.new_zeros(batch_size, 1)
            encoder_context = torch.cat((start_state, action.reshape(batch_size, -1), noise_magnitude), dim=1)
            env_code, _ = self.encoder.encode(image, encoder_context)
            context = torch.cat((start_state, action.reshape(batch_size, -1), env_code, noise_magnitude), dim=1)
            z_contact, log_prob = self.contact_prior(z=None, logpx=0, context=context, reverse=reverse)
            z_dynamic, log_prob = self.dynamic_prior(z=None, logpx=log_prob, context=context, reverse=reverse)
            x_contact, x_dynamic, log_prob = self.flow(z_contact, z_dynamic, logpx=log_prob, context=context, reverse=reverse)
            relative_displacement = x_dynamic.reshape(batch_size, -1, self.state_dim)
            traj = start_state.unsqueeze(1) + torch.cumsum(relative_displacement, dim=1)
            traj = traj * self.state_std + self.state_mean
            if not output_contact:
                return traj, log_prob
            else:
                contact_flag = (x_contact > 0).float()
                return traj, contact_flag, log_prob
        else:           # training
            assert traj is not None
            assert contact_flag is not None
            noise_magnitude = torch.rand((batch_size, 1), dtype=action.dtype, device=action.device) * 2 - 1     # (-1, 1)
            encoder_context = torch.cat((start_state, action.reshape(batch_size, -1), noise_magnitude), dim=1)
            env_code, _ = self.encoder.encode(image, encoder_context)
            context = torch.cat((start_state, action.reshape(batch_size, -1), env_code, noise_magnitude), dim=1)

            traj = (traj - self.state_mean) / self.state_std
            traj_noise = torch.randn(traj.shape, dtype=traj.dtype, device=traj.device) * noise_magnitude.abs().unsqueeze(-1)
            traj = traj + traj_noise

            contact_flag = (contact_flag - 0.5) / 0.5
            contact_noise = torch.randn(contact_flag.shape, dtype=contact_flag.dtype, device=contact_flag.device) * noise_magnitude.abs()
            contact_flag = contact_flag + contact_noise

            before_traj = torch.cat((start_state.unsqueeze(1), traj[:, :-1]), dim=1)
            relative_traj = traj - before_traj
            relative_traj = relative_traj.reshape(batch_size, -1)

            z_contact, z_dynamic, log_prob = self.flow(contact_flag, relative_traj, logpx=0, context=context, reverse=reverse)
            z_contact, log_prob = self.contact_prior(z=z_contact, logpx=log_prob, context=context, reverse=reverse)
            z_dynamic, log_prob = self.dynamic_prior(z=z_dynamic, logpx=log_prob, context=context, reverse=reverse)
            if not output_contact:
                return z_dynamic, log_prob
            else:
                return z_dynamic, z_contact, log_prob
