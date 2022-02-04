import torch
import colorednoise


class ICEM:

    def __init__(self, cost, dx, du, horizon, num_samples, num_elites, alpha, noise_beta, iterations, sigma=1.0,
                 elites_keep_fraction=0.3, control_constraints=None, device='cuda:0',
                 action_transform=None, flow=False):

        self.dx = dx
        self.du = du
        self.H = horizon
        self.N = num_samples
        self.K = num_elites
        self.alpha = alpha
        self.noise_beta = noise_beta
        self.iter = iterations
        self.keep_fraction = elites_keep_fraction
        self.sigma = sigma

        self.cost = cost
        self.control_constraints = control_constraints
        self.device = device

        # initialise mean and std of actions
        self.mean = torch.zeros(horizon, du, device=device)
        self.std = self.sigma * torch.ones(horizon, du, device=device)
        self.kept_elites = None
        self.action_transform = action_transform
        self.flow = flow

    def reset(self):
        self.mean = torch.zeros(self.H, self.du, device=self.device)
        self.std = self.sigma * torch.ones(self.H, self.du, device=self.device)
        self.kept_elites = None

    @property
    def best_K_U(self):
        return self.mean.unsqueeze(0) + self.std * torch.randn(self.K, self.H, self.du, device=self.device)

    def sample_action_sequences(self, state, N, sample_from_flow=False):
        # colored noise
        if self.noise_beta > 0 and not sample_from_flow:
            # Important improvement
            # self.mean has shape h,d: we need to swap d and h because temporal correlations are in last axis)
            # noinspection PyUnresolvedReferences
            samples = colorednoise.powerlaw_psd_gaussian(self.noise_beta, size=(N, self.du,
                                                                                self.H)).transpose(
                [0, 2, 1])
            samples = torch.from_numpy(samples).to(device=self.device).float()
        else:
            samples = torch.randn(N, self.H, self.du, device=self.device).float()

        if sample_from_flow:
            U = self.action_transform(state, samples)
        else:
            U = self.mean + self.std * samples

        if self.control_constraints is not None:
           U = torch.clamp(U, min=self.control_constraints[0], max=self.control_constraints[1])

        return U

    def update_distribution(self, elites):
        """
        param: elites - K x H x du number of best K control sequences by cost
        """

        # fit around mean of elites
        new_mean = elites.mean(dim=0)
        new_std = elites.std(dim=0)

        self.mean = (1 - self.alpha) * new_mean + self.alpha * self.mean  # [h,d]
        self.std = (1 - self.alpha) * new_std + self.alpha * self.std

    def step(self, x):

        # roll distirbution
        self.mean = torch.roll(self.mean, -1, dims=0)
        self.mean[-1] = torch.zeros(self.du, device=self.device)
        self.std = self.sigma * torch.ones(self.H, self.du, device=self.device)

        # Shift the keep elites
        # keep elites is 0.3*K, H, dx
        if self.kept_elites is not None:
            self.kept_elites = torch.roll(self.kept_elites, -1, dims=1)
            self.kept_elites[:, -1] = 0*torch.randn(len(self.kept_elites), self.du, device=self.device)

        # Generate new potential elites from flow
        #if self.flow:
        #    M = int(self.K * self.keep_fraction)
        #    if self.kept_elites is None:
        #        flow_elites = self.action_transform(x, torch.randn(M, self.H, self.du, device=self.device))
        #        self.kept_elites = flow_elites
        #    else:
        #        # We will just replace half the kept elites with flow elites -- maybe in future choose according to cost
        #        flow_elites = self.action_transform(x, torch.randn(M, self.H, self.du, device=self.device))
        #        self.kept_elites = torch.cat((self.kept_elites, flow_elites), dim=0)

        for i in range(self.iter):
            sample_from_flow = True if (i == 0) and self.flow else False
            if self.kept_elites is None:
                # Sample actions
                U = self.sample_action_sequences(x, self.N, sample_from_flow)
            else:
                # resuse the elites from the previous iteration
                U = self.sample_action_sequences(x, self.N-len(self.kept_elites), sample_from_flow)
                U = torch.cat((U, self.kept_elites), dim=0)

            # evaluate costs and update the distribution!
            costs = self.cost(x, U) # should be N costs
            sorted, indices = torch.sort(costs)
            elites = U[indices[:self.K]]

            self.update_distribution(elites)
            # save kept elites fraction
            self.kept_elites = U[indices[:int(self.K*self.keep_fraction)]]

        # Return the best action sequence - not the mean
        return U[indices[0]]
        #return self.mean






