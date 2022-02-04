import torch
from torch import optim

from flow_mpc.trainer import SVIMPC_LossFcn


class FlowController:

    def __init__(self, action_sampler, horizon, action_dim, state_dim, generative_model, N=100, device='cuda:0',
                 use_true_grad=False):
        self.generative_model = generative_model
        self.action_sampler = action_sampler
        self.du = action_dim
        self.H = horizon
        self.device = device
        self.sdf = torch.zeros((N, 64, 64), device=device)
        self.sdf_grad = self.sdf.clone()
        self.normalised_sdf = self.sdf.clone()
        self.z_env = None
        self.goal = torch.zeros((N, state_dim), device=device)
        self.N = N
        self.U = None
        self.imagine_sdf = False
        self.imagined_sdf = None
        self.use_true_grad = use_true_grad

    def update_environment(self, sdf, sdf_grad=None):
        self.sdf = torch.from_numpy(sdf).to(device=self.device).reshape(1, 1, 64, 64).float()
        self.normalised_sdf = torch.where(self.sdf < 0, self.sdf / 1000.0, self.sdf) + 0.02 * torch.randn(
            self.sdf.shape,
            device=self.device
        )
        self.sdf = torch.where(self.sdf < 0, self.sdf * 5, self.sdf)
        with torch.no_grad():
            self.z_env = self.action_sampler.encode_environment(self.normalised_sdf[0].unsqueeze(0))['z_environment']

        if sdf_grad is not None:
            self.sdf_grad = torch.from_numpy(sdf_grad).to(device=self.device).reshape(1, 64, 64, 2).float()
        else:
            self.sdf_grad = torch.empty(0)

    def update_goal(self, goal):
        self.goal = torch.from_numpy(goal).to(device=self.device).reshape(1, -1).float()

    def step(self, state):
        with torch.no_grad():
            tstate = torch.from_numpy(state).to(device=self.device).reshape(1, -1).float()
            action_sequence, _, _ = self.action_sampler.sample(tstate, self.goal, environment=None,
                                                               z_environment=self.z_env, N=self.N)

            u = action_sequence.reshape(self.N, self.H, self.du)

            log_pCost, log_pU, _ = self.generative_model.log_likelihood(tstate.repeat(self.N, 1),
                                                                        self.goal.repeat(self.N, 1),
                                                                        self.sdf.repeat(self.N, 1, 1,
                                                                                        1),
                                                                        None, u)
            log_likelihood = log_pCost + log_pU
            log_likelihood -= torch.mean(log_likelihood)
            log_likelihood /= torch.std(log_likelihood)
            u_weight = torch.softmax(1e3 * log_likelihood, dim=0)
            u = u.permute(1, 2, 0) @ u_weight

        # return first action from sequence
        self.U = u
        return u[0].cpu().numpy(), u.cpu().numpy()

    def load_model(self, model_path):
        self.action_sampler.load_state_dict(torch.load(model_path, map_location=self.device))

    def project_imagined_environment(self, state):

        loss_fn = SVIMPC_LossFcn(self.generative_model, False, use_grad=self.use_true_grad)

        num_iters = 500
        lr = 1e-2

        z_env = torch.nn.Parameter(self.z_env.clone())
        z_env.requires_grad = True
        optimiser = optim.Adam(
            [{'params': z_env}],
            lr=lr
        )

        # TODO Instead we will randomly sample starts and goals in the environments that are not in collision
        # tstate = torch.from_numpy(state).to(device=self.device).reshape(1, -1).float()
        num_planning_problems = 100
        num_samples = 100

        # Randomly sample starts and goals
        INVALID_VALUE = -99
        states = INVALID_VALUE * torch.ones(num_planning_problems - 1, 4, device=self.device)
        goals = INVALID_VALUE * torch.ones(num_planning_problems - 1, 4, device=self.device)

        while (states == INVALID_VALUE).sum() > 0 or (goals == INVALID_VALUE).sum() > 0:
            prospective_states = -1.8 + 3.6 * torch.rand(num_planning_problems - 1, 4, device=self.device)
            prospective_goals = -1.8 + 3.6 * torch.rand(num_planning_problems - 1, 4, device=self.device)
            g = torch.clamp(64 * (prospective_goals[:, :2] + 2) / 4, min=0, max=63).long()
            s = torch.clamp(64 * (prospective_states[:, :2] + 2) / 4, min=0, max=63).long()
            goals = torch.where(index_sdf(self.sdf.repeat(num_planning_problems - 1, 1, 1, 1), g) > 0,
                                prospective_goals, goals)
            states = torch.where(index_sdf(self.sdf.repeat(num_planning_problems - 1, 1, 1, 1), s) > 0,
                                 prospective_states,
                                 states)

        goals[:, 2:] = 0.0
        states[:, 2:] = torch.randn_like(states[:, 2:])

        goals = torch.cat((goals, self.goal.reshape(1, -1)), dim=0).float()
        states = torch.cat((states, torch.from_numpy(state).reshape(1, -1).to(device=self.device)), dim=0).float()

        # visualise_starts_and_goals(states.detach().cpu().numpy(), goals.detach().cpu().numpy(),
        #                           self.sdf.detach().cpu().numpy()[0, 0])

        if not self.use_true_grad:
            alpha, beta, gamma, kappa = 1.0, 0.01, 1.0, 1.0
        else:
            alpha = 1.0
            gamma = 5
            beta = 0.0
            kappa = 1.

        p_z_env = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        for iter in range(num_iters):
            U, log_qU, context_dict = self.action_sampler.sample(states, goals,
                                                                 environment=None,
                                                                 z_environment=z_env.repeat(
                                                                     num_planning_problems,
                                                                     1, 1, 1),
                                                                 N=num_samples)

            log_p_env = p_z_env.log_prob(z_env).sum(dim=[1, 2, 3])

            loss_dict, _ = loss_fn.compute_loss(U, log_qU, states, goals,
                                                self.sdf.repeat(num_planning_problems, 1, 1, 1),
                                                self.sdf_grad.repeat(num_planning_problems, 1, 1, 1),
                                                log_p_env,
                                                alpha, beta, gamma, kappa)

            loss_dict['total_loss'].backward()
            optimiser.step()
            optimiser.zero_grad()

        with torch.no_grad():
            imagined_sdf, _ = self.action_sampler.environment_encoder.reconstruct(z_env, N=1)
            imagined_sdf = imagined_sdf.cpu().numpy()[0, 0]
        self.imagined_sdf = imagined_sdf

        self.z_env = z_env


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


def visualise_trajectories(starts, goals, trajectories, sdf):
    import matplotlib.pyplot as plt
    import numpy as np
    from cv2 import resize
    fig, ax = plt.subplots(1, 1)
    big_sdf = resize(sdf, (256, 256))
    goals = np.clip(256 * (goals[:, :2] + 2) / 4, a_min=0, a_max=255).astype(np.uint64)
    starts = np.clip(256 * (starts[:, :2] + 2) / 4, a_min=0, a_max=255).astype(np.uint64)

    ax.imshow(big_sdf[::-1])
    for goal, start, trajs in zip(goals, starts, trajectories):
        plt.plot(goal[0], 255 - goal[1], marker='o', color="red", linewidth=2)
        plt.plot(start[0], 255 - start[1], marker='o', color="green", linewidth=2)
        # plt.plot([start[0], goal[0]], [255 - start[1], 255 - goal[1]], color='b', alpha=0.5)

        positions = trajs[:, :, :2]
        positions_idx = np.clip(256 * (positions + 2) / 4, a_min=0, a_max=255).astype(np.uint64)

        for i in range(len(positions_idx)):
            ax.plot(positions_idx[i, :, 0], 255 - positions_idx[i, :, 1], linewidth=2)
    plt.show()


def index_sdf(sdf, indices):
    device = sdf.device
    N = sdf.shape[0]
    indexy = torch.arange(64, device=device, dtype=torch.long).reshape(1, 64, 1).repeat(N, 1, 1)
    indexy[:, :, 0] = indices[:, 0].reshape(-1, 1).repeat(1, 64)
    y_indexed_sdf = sdf.view(-1, 64, 64).gather(2, indexy).squeeze(2)
    return y_indexed_sdf.gather(1, indices[:, 1].reshape(-1, 1))
