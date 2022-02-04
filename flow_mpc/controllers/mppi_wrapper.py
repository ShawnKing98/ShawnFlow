import torch
from torch import optim
from flow_mpc.controllers.mppi import MPPI
from flow_mpc.controllers.svmpc import SVMPC
from flow_mpc.controllers.icem import ICEM

from flow_mpc.trainer import SVIMPC_LossFcn

import matplotlib.pyplot as plt
from flow_mpc.visualisation import add_trajectory_to_axis
import matplotlib

matplotlib.use('tkAgg')


class FlowMPPI:

    def __init__(self, generative_model, horizon, action_dim, state_dim=4, control_constraints=None,
                 N=10,
                 device='cuda:0',
                 action_sampler=None,
                 sample_nominal=False,
                 sigma=1.0,
                 lambda_=1.0,
                 use_true_grad=False,
                 use_vae=True,
                 controller_type='mppi'):

        self.use_true_grad = use_true_grad
        self.sample_nominal = sample_nominal
        if sample_nominal and action_sampler is None:
            raise ValueError('Must provide action sampler for sampling nominal MPPI trajectory')

        self.action_sampler = action_sampler
        self.generative_model = generative_model
        sigma = sigma
        flow = False
        if action_sampler is not None and not sample_nominal:
            flow = True

        if 'mppi' in controller_type:
            self.controller = MPPI(self.cost, state_dim, action_dim, horizon, N, lambda_, sigma,
                                   control_constraints=control_constraints, device=device,
                                   action_transform=self.action_transform, flow=flow)
        elif 'svmpc' in controller_type:
            iters = 5
            num_particles = 4
            samples_per_particle = 256 # int(1000 / (num_particles * iters))
            self.controller = SVMPC(self.cost, state_dim, action_dim, horizon=horizon, num_particles=num_particles,
                                    samples_per_particle=samples_per_particle,
                                    lambda_=1, sigma=0.5, lr=1, iters=iters, device=device,
                                    action_transform=self.action_transform, flow=flow)
            N = num_particles * samples_per_particle
        elif 'icem' in controller_type:
            N = 200
            K = 30
            self.controller = ICEM(self.cost, state_dim, action_dim, horizon, N, K, 0.1, 2.5,
                                   sigma=0.5, elites_keep_fraction=0.3,
                                   iterations=5, control_constraints=None, device=device,
                                   action_transform=self.action_transform, flow=flow)

        self.dx = state_dim
        self.du = action_dim
        self.H = horizon
        self.device = device
        self.sdf = torch.zeros((1, 64, 64), device=device)
        self.goal = torch.zeros((1, state_dim), device=device)
        self.N = N
        self.use_vae = use_vae

    def cost(self, x, U):
        N, H, du = U.shape
        log_pCost, log_pU, X = self.generative_model(
            x.unsqueeze(0).repeat(1, N, 1),
            self.goal.unsqueeze(0).repeat(1, N, 1),
            self.sdf,
            None, U.unsqueeze(0))

        if False:
            fig, ax = plt.subplots()
            add_trajectory_to_axis(ax, x[0].detach().cpu().numpy(),
                                   self.goal[0].detach().cpu().numpy(),
                                   X[0].detach().cpu().numpy(),
                                   self.raw_sdf)
            plt.show()

        return -(log_pCost + log_pU).squeeze(0)

    def action_transform(self, x, Z, reverse=False, return_logprob=False):
        if self.action_sampler is None or self.sample_nominal:
            return Z
        with torch.no_grad():
            N = Z.shape[0]
            if reverse:
                U, logpU, _ = self.action_sampler.likelihood(Z.reshape(1, -1, self.H, self.du),
                                                             x.reshape(1, self.dx),
                                                             self.goal[0].reshape(1, self.dx),
                                                             environment=None,
                                                             z_environment=self.z_env[0].unsqueeze(0)
                                                             )

            else:
                U, logpU, _ = self.action_sampler.reconstruct(Z.reshape(1, N, self.H * self.du),
                                                              x.reshape(1, self.dx),
                                                              self.goal[0].reshape(1, self.dx),
                                                              environment=None,
                                                              z_environment=self.z_env[0].unsqueeze(0)
                                                              )

            if return_logprob:
                return U.reshape(N, self.H, self.du), logpU.reshape(-1)

        return U.reshape(N, self.H, self.du)

    def update_environment(self, sdf, sdf_grad=None):
        self.sdf = torch.from_numpy(sdf).to(device=self.device).unsqueeze(0).unsqueeze(1)
        self.raw_sdf = sdf
        self.normalised_sdf = torch.where(self.sdf < 0, self.sdf / 1000.0,
                                          self.sdf)  # + 0.02 * torch.randn_like(self.sdf)
        # self.sdf *= 2
        # torch.where(self.sdf < 0, self.sdf, self.sdf)
        # self.sdf = torch.where(self.sdf < 0, -1e4, 0.0)
        self.sdf = self.normalised_sdf
        # if self.action_sampler is not None:
        #    self.action_sampler.update_environment(sdf, sdf_grad)
        if self.action_sampler is not None:
            with torch.no_grad():
                _, self.z_env, self.z_mu, self.z_sigma = self.action_sampler.environment_encoder.vae(
                    self.normalised_sdf[0].unsqueeze(0))
        if sdf_grad is not None:
            self.sdf_grad = torch.from_numpy(sdf_grad).to(device=self.device).unsqueeze(0)
        else:
            self.sdf_grad = torch.empty(1, *self.sdf.shape[2:], len(self.sdf.shape[2:]))

    def update_goal(self, goal):
        self.goal = torch.from_numpy(goal).to(device=self.device).reshape(1, -1).float()

    def step(self, state, project=False):
        tstate = torch.from_numpy(state).to(device=self.device).reshape(1, -1).float()

        if project:
            self.project_imagined_environment(state, 1)
            #self.random_shooting_best_env(state)

        with torch.no_grad():
            if self.sample_nominal:
                N = 5
                U, _, _ = self.action_sampler.sample(
                    tstate,
                    self.goal[0].reshape(1, -1),
                    environment=None,
                    N=N - 1,
                    z_environment=self.z_env[0].unsqueeze(0)
                )

                U = torch.cat((U.reshape(-1, self.H, self.du), self.controller.U.reshape(1, self.H, self.du)), dim=0)
                log_pCost, log_pU, _ = self.generative_model(
                    tstate[-N:],
                    self.goal[-N:],
                    self.sdf[-N:],
                    None, U)
                log_likelihood = log_pCost + log_pU

                u_idx = torch.argmax(log_likelihood)
                # u_idx = torch.randperm(self.N)[0]

                u_sampled = U[u_idx]
                # if log_likelihood[u_idx] < log_likelihood[-1]:
                #    u_sampled = U[-1]

                self.controller.U = u_sampled

        tstate = torch.from_numpy(state).to(device=self.device).reshape(1, -1).float()

        # Step will sometimes fail due to some stupid numerics (bad samples) -- this is a hack but it seems to be rare
        # enough that it is OK
        U = self.controller.step(tstate)
        # return first action from sequence, and entire action sequence
        return U[0].detach().cpu().numpy(), self.controller.best_K_U.detach().cpu().numpy(), U.detach().cpu().numpy()

    def random_shooting_best_env(self, state):
        num_envs = 100
        samples_per_env = 10
        z_env_dim = self.z_env.shape[-1]

        with torch.no_grad():
            starts = torch.from_numpy(state).reshape(1, -1).repeat(num_envs, 1).to(device=self.device).float()
            goals = self.goal.reshape(1, -1).repeat(num_envs, 1)

            # Sample a bunch of different environments
            z_env = self.action_sampler.environment_encoder.vae.prior.sample(sample_shape=(num_envs, z_env_dim))
            z_env[0] = self.z_env

            # Sample a bunch of trajectories for each goal
            U, log_qU, context_dict = self.action_sampler.sample(starts, goals, environment=None,
                                                                 z_environment=z_env, N=samples_per_env)

            # MPPI like thing for environment

            # Roll all envs together
            U = U.reshape(-1, self.H, self.du)

            # Evaluate costs            self.z_env = z_env[torch.argmin(costs)].unsqueeze(0)

            costs = self.cost(starts[0].unsqueeze(0), U).reshape(num_envs, samples_per_env).mean(dim=1)
            weights = torch.softmax(-costs/100, dim=0)
            self.z_env = torch.sum(weights.reshape(-1, 1) * z_env, dim=0, keepdim=True)

    def project_imagined_environment(self, state, num_iters, name=None):
        loss_fn = SVIMPC_LossFcn(self.generative_model, False, use_grad=self.use_true_grad)
        lr = 1e-2

        z_env = torch.nn.Parameter(self.z_env[0].unsqueeze(0).clone())
        z_env.requires_grad = True
        optimiser = optim.Adam(
            [{'params': z_env}],
            lr=lr
        )

        # TODO Instead we will randomly sample starts and goals in the environments that are not in collision
        # tstate = torch.from_numpy(state).to(device=self.device).reshape(1, -1).float()
        if num_iters > 10:
            num_planning_problems = 10
            num_samples = 100
        else:
            num_planning_problems = 1
            num_samples = 1000

        # Randomly sample starts and goals
        INVALID_VALUE = -99
        states = INVALID_VALUE * torch.ones(num_planning_problems, self.dx, device=self.device)
        goals = INVALID_VALUE * torch.ones(num_planning_problems, self.dx, device=self.device)

        if num_planning_problems > 1:
            while (states == INVALID_VALUE).sum() > 0 or (goals == INVALID_VALUE).sum() > 0:
                prospective_states = -1.8 + 3.6 * torch.rand(num_planning_problems, 4, device=self.device)
                prospective_goals = -1.8 + 3.6 * torch.rand(num_planning_problems, 4, device=self.device)
                g = torch.clamp(64 * (prospective_goals[:, :2] + 2) / 4, min=0, max=63).long()
                s = torch.clamp(64 * (prospective_states[:, :2] + 2) / 4, min=0, max=63).long()
                goals = torch.where(index_sdf(self.sdf[0].repeat(num_planning_problems, 1, 1, 1), g) > -1e-3,
                                    prospective_goals, goals)
                states = torch.where(index_sdf(self.sdf[0].repeat(num_planning_problems, 1, 1, 1), s) > -1e-3,
                                     prospective_states,
                                     states)

            goals[:, self.dx // 2:] = 0.0
            states[:, self.dx // 2:] = torch.randn_like(states[:, self.dx // 2:])
        goals[0] = self.goal[0]
        states[0] = torch.from_numpy(state).to(device=self.device).float()

        # visualise_starts_and_goals(states.detach().cpu().numpy(), goals.detach().cpu().numpy(),
        #                           self.sdf.detach().cpu().numpy()[0, 0])

        if name is not None:
            U, log_qU, context_dict = self.action_sampler.sample(states[0].unsqueeze(0), goals[0].unsqueeze(0),
                                                                 environment=self.normalised_sdf[0].unsqueeze(0),
                                                                 N=100)
            _, _, X = self.generative_model(
                states[0].repeat(100, 1).unsqueeze(0),
                goals[0].repeat(100, 1).unsqueeze(0),
                self.sdf[0].unsqueeze(0),
                None, U)

            visualise_trajectories(states[0].detach().unsqueeze(0).cpu().numpy(),
                                   goals[0].detach().unsqueeze(0).cpu().numpy(),
                                   X.detach().cpu().unsqueeze(0).numpy(), self.raw_sdf, f'{name}_before.png')

        if not self.use_true_grad:
            alpha, beta, gamma, kappa = 1.0, 1, 1.0, 0.001 * 64*64*64
            sigma = 0.25
        else:
            alpha = 1.0
            gamma = 5
            beta = 0.0
            kappa = 1.
            sigma = None

        p_z_env = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

        for iter in range(num_iters):

            if self.use_true_grad:
                U, log_qU, context_dict = self.action_sampler.sample(states, goals,
                                                                     environment=None,
                                                                     z_environment=z_env,
                                                                     N=num_samples)
            else:
                U, log_qU, context_dict = self.action_sampler.sample_w_peturbation(states,
                                                                                   goals,
                                                                                   environment=None,
                                                                                   z_environment=z_env,
                                                                                   N=num_samples,
                                                                                   sigma=sigma)

            if self.use_vae:
                log_p_env = self.action_sampler.environment_encoder.vae.prior.log_prob(z_env).sum(dim=1)
            else:
                log_p_env = p_z_env.log_prob(z_env).sum(dim=[1, 2, 3])
                log_p_env = torch.clamp(log_p_env, max=-2900)

            loss_dict, _ = loss_fn.compute_loss(U, log_qU, states, goals,
                                                self.sdf[0],
                                                self.sdf_grad[0],
                                                log_p_env, None,
                                                alpha, beta, gamma, kappa)

            loss_dict['total_loss'].backward()
            #print(log_p_env)
            optimiser.step()
            optimiser.zero_grad()

        #with torch.no_grad():
        #    imagined_sdf = self.action_sampler.environment_encoder.reconstruct(z_env, N=1)
        #    imagined_sdf = imagined_sdf['environments']
        #    imagined_sdf = imagined_sdf.cpu().numpy()[0, 0, 0]

        #self.imagined_sdf = imagined_sdf
        self.z_env = z_env

        if name is not None:
            U, log_qU, context_dict = self.action_sampler.sample(states[0].unsqueeze(0), goals[0].unsqueeze(0),
                                                                 environment=None,
                                                                 z_environment=z_env,
                                                                 N=100)
            _, _, X = self.generative_model.posterior_log_likelihood(
                states[0].repeat(100, 1).unsqueeze(0),
                goals[0].repeat(100, 1).unsqueeze(0),
                self.sdf[0].unsqueeze(0),
                None, U)

            visualise_trajectories(states[0].detach().unsqueeze(0).cpu().numpy(),
                                   goals[0].detach().unsqueeze(0).cpu().numpy(),
                                   X.detach().cpu().unsqueeze(0).numpy(), self.raw_sdf, f'{name}_after.png')

    def load_model(self, model_path):
        self.action_sampler.load_state_dict(torch.load(model_path, map_location=self.device))


def visualise_starts_and_goals(starts, goals, sdf):
    import matplotlib.pyplot as plt
    from cv2 import resize
    import numpy as np
    fig, ax = plt.subplots(1, 1)
    big_sdf = resize(sdf, (256, 256))
    goals = np.clip(256 * (goals[:, :2] + 2) / 4, a_min=0, a_max=255).astype(np.uint64)
    starts = np.clip(256 * (starts[:, :2] + 2) / 4, a_min=0, a_max=255).astype(np.uint64)

    ax.imshow(big_sdf[::-1])

    for goal, start in zip(goals, starts):
        plt.plot(goal[0], 255 - goal[1], marker='o', color="red", linewidth=2)
        plt.plot(start[0], 255 - start[1], marker='o', color="green", linewidth=2)
        plt.plot([start[0], goal[0]], [255 - start[1], 255 - goal[1]], color='b', alpha=0.5)

    plt.show()


def visualise_trajectories(starts, goals, trajectories, sdf, name):
    import matplotlib.pyplot as plt
    import numpy as np
    from cv2 import resize
    fig, ax = plt.subplots(1, 1)
    big_sdf = resize(sdf, (256, 256))
    goals = np.clip(256 * (goals[:, :2] + 2) / 4, a_min=0, a_max=255).astype(np.uint64)
    starts = np.clip(256 * (starts[:, :2] + 2) / 4, a_min=0, a_max=255).astype(np.uint64)

    ax.imshow(big_sdf[::-1])
    for goal, start, trajs in zip(goals, starts, trajectories):
        plt.plot(goal[0], 255 - goal[1], marker='x', color="red", linewidth=2)
        plt.plot(start[0], 255 - start[1], marker='o', color="green", linewidth=2)
        # plt.plot([start[0], goal[0]], [255 - start[1], 255 - goal[1]], color='b', alpha=0.5)
        positions = trajs[:, :, :2]
        positions_idx = np.clip(256 * (positions + 2) / 4, a_min=0, a_max=255).astype(np.uint64)

        for i in range(len(positions_idx)):
            ax.plot(positions_idx[i, :, 0], 255 - positions_idx[i, :, 1], linewidth=1, alpha=0.5)

    fig.savefig(name)


def index_sdf(sdf, indices):
    device = sdf.device
    N = sdf.shape[0]
    indexy = torch.arange(64, device=device, dtype=torch.long).reshape(1, 64, 1).repeat(N, 1, 1)
    indexy[:, :, 0] = indices[:, 0].reshape(-1, 1).repeat(1, 64)
    y_indexed_sdf = sdf.view(-1, 64, 64).gather(2, indexy).squeeze(2)
    return y_indexed_sdf.gather(1, indices[:, 1].reshape(-1, 1))
