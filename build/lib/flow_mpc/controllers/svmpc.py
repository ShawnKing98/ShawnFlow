import torch

from gpytorch.kernels import RBFKernel
import torch.distributions as dist
from torch import optim
from KDEpy import bw_selection


def bw_median(x: torch.Tensor, y: torch.Tensor = None, bw_scale: float = 1.0, tol: float = 1.0e-5) -> torch.Tensor:
    if y is None:
        y = x.detach().clone()
    pairwise_dists = squared_distance(x, y).detach()
    h = torch.median(pairwise_dists)
    # TODO: double check which is correct
    # h = torch.sqrt(0.5 * h / torch.tensor(x.shape[0] + 1)).log()
    h = torch.sqrt(0.5 * h) / torch.tensor(x.shape[0] + 1.0).log()
    return bw_scale * h.clamp_min_(tol)
    i


def squared_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Computes squared distance matrix between two arrays of row vectors.
    Code originally from:
        https://github.com/pytorch/pytorch/issues/15253#issuecomment-491467128
    """
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(
        x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
    ).add_(x1_norm)
    return res.clamp(min=0)  # avoid negative distances due to num precision


def get_gmm(means, weights, scale):
    mix = dist.Categorical(weights)
    comp = dist.Independent(dist.Normal(loc=means.detach(), scale=scale), 1)
    return dist.mixture_same_family.MixtureSameFamily(mix, comp)


def d_log_qU(U_samples, U_mean, U_sigma):
    return (U_samples - U_mean) / U_sigma ** 2


class SVMPC:

    def __init__(self, cost, dx, du, horizon, num_particles, samples_per_particle,
                 lr=0.01, lambda_=1.0, sigma=1.0, iters=1,
                 control_constraints=None,
                 device='cuda:0', action_transform=None, flow=False):
        """

            SVMPC
            optionally uses flow to represent posterior q(U) --
            samples maintained in U space. q(U) is mixture of Gaussians, in Flow space

        """

        self.dx = dx
        self.du = du
        self.H = horizon
        self.cost = cost
        self.control_constraints = control_constraints
        self.device = device
        self.M = num_particles
        self.N = samples_per_particle
        self.lr = lr
        self.kernel = RBFKernel().to(device=device)
        self.lambda_ = lambda_
        self.sigma = sigma
        self.iters = iters
        self.warmed_up = False
        self.action_transform = action_transform
        self.flow = flow

        # sample initial actions
        self.U = torch.randn(self.M, self.H, self.du, device=self.device)
        self.U.requires_grad = True

        self.weights = torch.ones(self.M, device=self.device) / self.M

        self.reset()

    @property
    def best_K_U(self):
        return self.U.detach()

    def step(self, state):
        if self.warmed_up:
            for _ in range(self.iters):
                self.update_particles(state)
        else:
            with torch.no_grad():
                if self.flow:
                    self.U = self.action_transform(state, self.U)
            for _ in range(25):
                self.update_particles(state)

            self.warmed_up = True

        with torch.no_grad():
            if self.flow:
                U = self.action_transform(state, torch.randn(self.M, self.H, self.du, device=self.device))
                U = torch.cat((self.U, U), dim=0)
                costs = self.cost(state, U)
                weights = torch.softmax(-costs / self.lambda_, dim=0)
            else:
                # compute costs
                costs = self.cost(state, self.U)

                # Update weights
                weights = torch.softmax(-costs / self.lambda_, dim=0)
                U = self.U

            # Get out U
            out_U = U[torch.argmax(weights)]

            self.U = U[torch.argsort(weights, descending=True)[:self.M]]
            self.weights = U[torch.argsort(weights, descending=True)[:self.M]]

            # shift actions & psterior
            self.U = torch.roll(self.U, -1, dims=1)
            self.U[:, -1] = 0*self.sigma * torch.randn(self.M, self.du, device=self.device)

        return out_U.detach()

    def update_particles(self, state):
        # first N actions from each mixture
        with torch.no_grad():
            noise = torch.randn(self.N, self.M, self.H, self.du, device=self.device)
            if False:#"#self.flow:
                Z_noise = noise[:self.N//2].reshape(-1, self.H, self.du)
                U_noise = self.sigma * noise[self.N//2:]
                U_samples_flow = self.action_transform(state, Z_noise).reshape(self.N//2, self.M, self.H, self.du)
                U_samples_gaussian = self.U.unsqueeze(0) + U_noise
                U_samples = torch.cat((U_samples_flow, U_samples_gaussian), dim=0)
            else:
                U_samples = self.sigma*noise + self.U.unsqueeze(0)

            # Evaluate cost of action samples - evaluate each action set N times
            costs = self.cost(state, U_samples.reshape(-1, self.H, self.du)).reshape(self.N, self.M)
            weights = torch.softmax(-costs / self.lambda_, dim=0)

            bw = bw_median(self.U.flatten(1, -1), self.U.flatten(1, -1), 1)
            self.kernel.lengthscale = bw

        self.U.requires_grad = True
        # add kernel
        Kxx = self.kernel(
            self.U.flatten(1, -1), self.U.clone().detach().flatten(1, -1)
        ).evaluate()
        grad_k = torch.autograd.grad(Kxx.sum(), self.U)[0]

        with torch.no_grad():
            grad_lik = d_log_qU(U_samples, self.U.unsqueeze(0), self.sigma)
            grad_lik = (weights.reshape(self.N, self.M, 1, 1) * grad_lik).sum(dim=0)
            phi = grad_k + torch.tensordot(Kxx, grad_lik, 1) / self.M

            self.U = self.U + self.lr * phi

    def reset(self):
        # sample initial actions
        self.U = torch.randn_like(self.U)
        self.U.requires_grad = True

        self.warmed_up = False
        self.weights = torch.ones(self.M, device=self.device) / self.M