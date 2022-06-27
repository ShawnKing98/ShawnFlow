import torch


class MPPI:

    def __init__(self, cost, dx, du, horizon, num_samples, lambda_, sigma, control_constraints=None,
                 device='cuda:0', action_transform=None, flow=False):
        self.dx = dx
        self.du = du
        self.cost = cost
        self.H = horizon
        self.N = num_samples
        self.sigma = sigma
        self.lambda_ = lambda_
        self.device = device
        self.control_constraints = control_constraints
        self.U = self.sigma * torch.randn(self.H, self.du, device=self.device)
        self.Z = self.U
        self.action_transform = action_transform
        self.flow = flow
        self.best_K_U = None
        self.K = 10

    def step(self, x):

        # Sample peturbations
        noise = torch.randn(self.N, self.H, self.du, device=self.device)
        noise[-1] *= 0.0
        # Get peturbation cost
        peturbed_actions = torch.zeros_like(noise)

        if self.flow:
            # Peturbed actions in action space
            peturbed_actions[:self.N//2] = self.action_transform(x[0].unsqueeze(0), noise[:self.N//2])
            peturbed_actions[self.N//2:] = self.U.unsqueeze(dim=0) + self.sigma * noise[self.N//2:]

            action_cost_Z = torch.sum(0.1 * (noise - self.Z) ** 2, dim=[1, 2])
            action_cost_U = torch.sum(self.lambda_ * noise * self.U / self.sigma, dim=[1, 2])

            action_cost = torch.cat((action_cost_Z[:self.N//2], action_cost_U[:self.N//2]), dim=0) / self.du

        else:
            peturbed_actions = self.U.unsqueeze(dim=0) + self.sigma * noise
            if self.control_constraints is not None:
                peturbed_actions = torch.clamp(peturbed_actions, min=self.control_constraints[0],
                                               max=self.control_constraints[1])
            action_cost = torch.sum(self.lambda_ * noise * self.U / self.sigma, dim=[1, 2]) / self.du
        # peturbed_actions = 10 * torch.ones_like(peturbed_actions)
        # Get total cost
        total_cost = self.cost(x, peturbed_actions)
        # total_cost -= torch.min(total_cost)
        total_cost += action_cost
        # omega = torch.exp(-total_cost / self.lambda_)
        # omega /= torch.sum(omega)
        total_cost -= total_cost.min()
        omega = torch.softmax(-total_cost / self.lambda_, dim=0)

        self.U = torch.sum((omega.reshape(-1, 1, 1) * peturbed_actions), dim=0)
        _, idx = torch.sort(total_cost, dim=0, descending=False)
        self.best_K_U = torch.gather(peturbed_actions, 0, idx[:self.K].reshape(-1, 1, 1).repeat(1, self.H, self.du))

        #
        ## Shift U along by 1 timestep
        #if self.action_transform is not None:
        #    # We transform to true action space, do the shifting, then transform #back
        #    Z = self.action_transform(x[0].unsqueeze(0),
        #                              Z.unsqueeze(0))[0]
        #    self.U = torch.roll(Z, -1, dims=0)
        #    self.U[-1] = torch.randn(self.du, device=self.device)
        #
        #    # Save shifted U as the nominal trajectory for next time
        #    self.Z = self.action_transform(x[0].unsqueeze(0),
        #                                   self.U.unsqueeze(0),
        #                                   reverse=True)[0]
        #    # We return the unshifted version
        #    return Z

        out_U = self.U.clone()
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = torch.zeros(self.du, device=self.device)

        if self.action_transform is not None:
            self.Z = self.action_transform(x[0].unsqueeze(0),
                                           self.U.unsqueeze(0),
                                           reverse=True)

        return out_U

    def reset(self):
        self.U = self.sigma * torch.randn(self.H, self.du, device=self.device)
