import torch
from torch import nn
from flow_mpc.models.double_integrator import trajectory_kernel
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from flow_mpc.visualisation import add_trajectory_to_axis

EPSILON = 1e-6

import time


class Trainer(nn.Module):

    def __init__(self, planning_network, loss_fn):
        super().__init__()

        self.planning_network = planning_network
        self.loss_fn = loss_fn
        self.metadata = None

    def forward(self, starts, goals, sdf, sdf_grad, U,
                samples_per_env, alpha, beta, gamma, kappa, sigma=None, plot=False, reconstruct=True):
        if U is None:
            U, log_qU, context_dict = self.planning_network(starts, goals, sdf,
                                                            samples_per_env, sigma=sigma, reconstruct=reconstruct)
        else:
            _, log_qU, context_dict = self.planning_network.likelihood(U.unsqueeze(1), starts, goals, sdf, reconstruct=reconstruct)

        loss_dict, metadata = self.loss_fn.compute_loss(U, log_qU, starts, goals, sdf, sdf_grad,
                                                        context_dict['log_p_env'], context_dict['reg'],
                                                        alpha, beta, gamma, kappa,
                                                        prior_weights=None, plot=plot)

        self.metadata = metadata
        return loss_dict


class SVIMPC_LossFcn:

    def __init__(self, cost_model, repel_trajectories=False, use_grad=True, supervised=False):
        self.cost_model = cost_model
        self.repel_trajectories = repel_trajectories
        self.use_grad = use_grad
        self.supervised = supervised

    def compute_loss(self, U, log_qU, starts, goals, environments, environment_grad, log_p_env, regularisation,
                     alpha=1, beta=1, gamma=1, kappa=1, prior_weights=None, plot=False):
        """
            Computes loss with or without gradients, depending on class variable use_grad
               :param U: Action sequence
               :param log_qU: Likelihood of action sequence under approximate posterior
               :param starts: start states for planning
               :param goals: goal states for planning
               :param environments: Environment SDF
               :param environment_grad: Environment Gradient SDF
               :param log_p_env: log likelihood of environments SDF under generative model of environments (flow)
               :param alpha: Dependent on if grad or not is used
               :param beta: Dependent on if grad or not is used
               :param gamma: Dependent on if grad or not is used
               :param kappa: Controls strength of loss on generative model of environments. High kappa -> more weight on modelling environments
               :return: Scalar loss value
        """

        # Computes average loss per environments
        if self.use_grad:
            loss, metadata = self.grad_loss(U, log_qU, starts, goals, environments,
                                            environment_grad, alpha, beta, gamma)
        elif self.supervised:
            loss, metadata = self.supervised_loss(U, log_qU, starts, goals, environments,
                                                  environment_grad, alpha, beta, gamma, plot=plot)
        else:
            loss, metadata = self.grad_free_loss(U, log_qU, starts, goals,
                                                 environments, alpha, beta, gamma, prior_weights=prior_weights,
                                                 plot=plot)

        # Environment on loss
        if log_p_env is not None:
            loss['total_loss'] = loss['total_loss'] - kappa * log_p_env / np.prod(environments.shape[1:])
            loss['log_p_env'] = log_p_env.detach().mean()

        loss['total_loss'] = loss['total_loss'].mean()

        if regularisation is not None:
            loss['total_loss'] = loss['total_loss'] + 0.01 * regularisation[0].abs().mean() + 0.01 * regularisation[
                1].abs().mean()

        return loss, metadata

    def supervised_loss(self, U, log_qU, starts, goals, environments, environment_grad, alpha=1, beta=1, gamma=1, plot=False):
        loss = {
            'total_loss': -log_qU.mean(),
            'log_p_cost': torch.tensor(0.),
            'log_p_U': torch.tensor(0.),
            'log_q_U': torch.tensor(0.),
        }
        B, H, du = U.shape
        metadata = {}


        if plot:
            with torch.no_grad():
                log_p_cost, log_pU, X = self.cost_model(
                    starts.unsqueeze(1),
                    goals.unsqueeze(1),
                    environments,
                    None,
                    U.unsqueeze(1)
                )

            fig, axes = plt.subplots(4, 4)
            axes = axes.flatten()

            for i, ax in enumerate(axes):
                add_trajectory_to_axis(ax, starts[i].detach().cpu().numpy(),
                                       goals[i].detach().cpu().numpy(),
                                       X[i].detach().cpu().numpy(),
                                       environments[i, 0].detach().cpu().numpy())

            metadata['Training trajectories'] = {
                'type': 'figure',
                'data': fig
            }

        return loss, metadata

    def grad_loss(self, U, log_qU, starts, goals, environments, environment_grad, alpha=1, beta=1, gamma=1):
        """
        Computes loss using gradient through cost function and dynamics
        :param U: Action sequence
        :param log_qU: Likelihood of action sequence under approximate posterior
        :param starts: start states for planning
        :param goals: goal states for planning
        :param environments: Environment SDF
        :param environment_grad: Environment Gradient SDF
        :param alpha: Controls weight of trajectory cost on loss -> low alpha favours entropy of approx. distribution
        :param beta: Controls weight on trajectory kernel -> high beta 'pushes' trajectories away from one another
        :param gamma: Controls weight on collision cost -> high gamma makes collision for costly
        :return: Loss which is size (B) where B is the batch size (i.e. number of environments)
        """
        metadata = {}
        B, N, H, du = U.shape
        _, dx = starts.shape
        log_p_cost, log_pU, X = self.cost_model(
            starts.unsqueeze(1).repeat(1, N, 1),
            goals.unsqueeze(1).repeat(1, N, 1),
            gamma * environments,
            gamma * environment_grad,
            U)

        # reshape with samples per environments
        log_p_cost = log_p_cost.reshape(B, N)
        log_pU = log_pU.reshape(B, N)
        X = X.reshape(B, N, H, dx)

        free_energy = (log_qU - alpha * (log_p_cost + log_pU)).mean(dim=1)

        loss = {
            'total_loss': free_energy,
            'log_p_cost': log_p_cost.detach().mean(),
            'log_p_U': log_pU.detach().mean(),
            'log_q_U': log_qU.detach().mean(),
        }

        # TODO this is broken since swapping (N, B) to (B, N)
        if self.repel_trajectories:
            K = trajectory_kernel(X[:, :, :, :2].permute(1, 2, 0, 3).contiguous(),
                                  goals.repeat(N, 1, 1).reshape(N * B, dx))
            free_energy += beta * K
            loss['Kxx'] = K.detach().mean()

        return loss, metadata

    def grad_free_loss(self, U, log_qU, starts, goals, environments, alpha=1, beta=1, gamma=1, prior_weights=None,
                       plot=False,
                       use_VIMPC_loss=True):
        """
             Computes loss without taking gradients through cost and dynamics. Does sample-based gradient estimation
             :param U: Action sequence
             :param log_qU: Likelihood of action sequence under approximate posterior
             :param starts: start states for planning
             :param goals: goal states for planning
             :param environments: Environment SDF
             :param environment_grad: Environment Gradient SDF
             :param alpha: Controls weight of prior on actions -> low alpha means deviations from action prior penalised less
             :param beta: Controls how trajectory cost effects sample weights -> beta->infinity sample weights selection -> max operator
             :param gamma: Controls how large an entropy bonus is given to the sample weights.
                            Least likely trajectory recieves a bonus of e^k to it's weight. High gamma -> more entropy
             :return: Loss which is size (B) where B is the batch size (i.e. number of environments)
             """
        B, N, H, du = U.shape
        _, dx = starts.shape
        metadata = {}

        s = time.time()
        with torch.no_grad():
            log_p_cost, log_pU, X = self.cost_model(
                starts.unsqueeze(1).repeat(1, N, 1),
                goals.unsqueeze(1).repeat(1, N, 1),
                environments,
                None,
                U
            )
        e = time.time()

        # Reshape likelihoods
        log_p_cost = log_p_cost.reshape(B, N)
        log_pU = log_pU.reshape(B, N)
        ll = log_p_cost + alpha * log_pU
        # just test something out.
        # ll = torch.clamp(ll, min=-20000)

        ll_range = torch.max(ll, dim=1, keepdim=True).values - torch.min(ll, dim=1, keepdim=True).values
        # normalised_ll = (ll - torch.max(ll, dim=1, keepdim=True).values) / (ll_range + EPSILON)
        normalised_ll = (ll - torch.max(ll, dim=1, keepdim=True).values)  # / 5000.0

        log_qU_range = torch.max(log_qU, dim=1, keepdim=True).values - torch.min(log_qU, dim=1, keepdim=True).values
        normalised_log_qU = (log_qU - torch.max(log_qU, dim=1, keepdim=True).values) / (log_qU_range + EPSILON)

        if use_VIMPC_loss:
            # use loss from VIMPC paper:
            # https://arxiv.org/abs/1907.04202
            # Compute sample weights

            sample_weights = torch.softmax(ll / beta - gamma * normalised_log_qU, dim=1).detach()
            negative_weights = torch.softmax(-ll / 1000.0, dim=1).detach()

            if prior_weights is not None:
                sample_weights *= prior_weights

            # negative_weights = (-normalised_ll / 0.01).exp()
            # negative_weights = negative_weights / negative_weights.sum(dim=0)

            # total_loss = ((negative_weights - sample_weights) * log_qU).sum(dim=100)

            # Loss
            # current_weights = log_qU - torch.max(log_qU, dim=0, keepdim=True).values

            current_weights = (1.0 + normalised_log_qU).detach()
            current_weights /= torch.sum(current_weights, dim=1, keepdim=True)

            total_loss = ((- sample_weights) * log_qU).sum(dim=1)

            # total_loss = 1000 * (sample_weights - current_weights).abs().sum(dim=0)
            # total_loss = - (sample_weights * log_qU).sum(dim=0)
        else:
            # Just use loss similar to standard policy gradients
            # Have normalised log_ll [-1, 0]
            # print(torch.max(ll, dim=1).values)
            # print(torch.max(torch.exp(ll / 100), dim=1).values)
            # r = torch.exp(normalised_ll / 10)
            # r = torch.exp(ll / 100)
            r = torch.exp(normalised_ll / beta)
            # r = normalised_ll
            # Have normalised log_qu [-1  0]
            lqu = normalised_log_qU
            lqu = log_qU
            sample_weights = r / torch.sum(r, dim=-1, keepdim=True)
            sample_weights = sample_weights.detach()
            current_weights = (1.0 + normalised_log_qU).detach()
            current_weights /= torch.sum(current_weights, dim=1, keepdim=True)
            # Do loss
            total_loss = torch.sum(-sample_weights * lqu, dim=-1) + gamma * torch.mean(lqu, dim=-1)
        loss = {
            'total_loss': total_loss,
            'log_p_cost': log_p_cost.detach().mean(),
            'log_p_U': log_pU.detach().mean(),
            'log_q_U': log_qU.detach().mean(),
        }

        if plot:
            # Plotting for debugging, send to tensorboard for first 16 environments, order by cost

            _, idx = torch.sort(normalised_ll, dim=1, descending=True)
            best_N = 100
            X = X.reshape(-1, N, H, dx)
            best_X = torch.gather(X[:16], 1,
                                  idx[:16, :best_N].reshape(16, best_N, 1, 1).repeat(1, 1, H, dx))
            fig, axes = plt.subplots(4, 4)
            axes = axes.flatten()
            for i, ax in enumerate(axes):
                add_trajectory_to_axis(ax, starts[i].detach().cpu().numpy(),
                                       goals[i].detach().cpu().numpy(),
                                       best_X[i].detach().cpu().numpy(),
                                       environments[i, 0].detach().cpu().numpy())

            metadata['best_sampled_trajectories'] = {
                'type': 'figure',
                'data': fig
            }

            fig2, axes = plt.subplots(4, 4)
            fig2, axes = plt.subplots(4, 4, figsize=(16, 16))
            fig2.tight_layout()
            best_W = torch.gather(sample_weights[:16], 1, idx[:16, :best_N])
            best_current_W = torch.gather(current_weights[:16], 1, idx[:16, :best_N])

            axes = axes.flatten()

            for i, ax in enumerate(axes):
                ax.hist(best_W[i].cpu().numpy(), color='b', alpha=0.5)
                ax.hist(best_current_W[i].cpu().numpy(), color='r', alpha=0.5)
            metadata['best_sampled_weights'] = {
                'type': 'figure',
                'data': fig2
            }

            fig4, axes = plt.subplots(4, 4, figsize=(16, 16))
            fig4.tight_layout()
            best_W = torch.gather(sample_weights[:16], 1, idx[:16, -best_N:])
            best_current_W = torch.gather(negative_weights[:16], 1, idx[:16, -best_N:])

            axes = axes.flatten()

            for i, ax in enumerate(axes):
                ax.hist(best_W[i].cpu().numpy(), color='b', alpha=0.5)
                ax.hist(best_current_W[i].cpu().numpy(), color='r', alpha=0.5)
            metadata['worst_sampled_weights'] = {
                'type': 'figure',
                'data': fig4
            }

            fig3, axes = plt.subplots(4, 4, figsize=(16, 16))
            fig3.tight_layout()
            axes = axes.flatten()
            for i, ax in enumerate(axes):
                ax.hist(sample_weights[i].detach().cpu().numpy(), color='b', alpha=0.5)
                ax.hist(current_weights[i].detach().cpu().numpy(), color='r', alpha=0.5)

            metadata['target_and_actual_weights'] = {
                'type': 'figure',
                'data': fig3
            }

        # Add some stuff we want to histogram for debugging
        # Note -- we collapse the number of samples and number of envs together for the metadata - as it all goes into
        # one big histogram, this is a lot of data so may change in future
        metadata['sample_weights'] = {
            'type': 'histogram',
            'data': sample_weights.detach().cpu().numpy().reshape(-1)
        }
        metadata['normalised_ll'] = {
            'type': 'histogram',
            'data': normalised_ll.detach().cpu().numpy().reshape(-1)
        }
        metadata['ll'] = {
            'type': 'histogram',
            'data': ll.detach().cpu().numpy().reshape(-1)
        }
        return loss, metadata
