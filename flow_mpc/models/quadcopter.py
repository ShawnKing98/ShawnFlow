import torch
import numpy as np
from torch import nn
from flow_mpc.models.generative_model import GenerativeModel

from flow_mpc.models.pytorch_transforms import euler_angles_to_matrix
from flow_mpc.models.utils import PointGoalFcn, CollisionFcn

class Quadcopter6DDynamics(nn.Module):

    def __init__(self, dt):
        super().__init__()
        self.dt = dt

    def forward(self, state, control):
        # Unroll state
        x, y, z, phi, theta, psi = torch.chunk(state, chunks=6, dim=-1)

        # Trigonometric fcns on all the angles needed for dynamics
        cphi = torch.cos(phi)
        ctheta = torch.cos(theta)
        cpsi = torch.cos(psi)

        sphi = torch.sin(phi)
        stheta = torch.sin(theta)
        spsi = torch.sin(psi)

        ttheta = torch.tan(theta)

        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)

        f1 = torch.cat((ctheta * cpsi, -cphi * spsi + cpsi * sphi * stheta,
                        spsi * sphi + cphi * cpsi * stheta, zeros, zeros, zeros), dim=1)


        f2 = torch.cat([ctheta * spsi, cphi * cpsi + sphi * spsi * stheta, -cpsi * sphi + cphi * spsi * stheta,
                        zeros, zeros, zeros], dim=1)
        f3 = torch.cat([-stheta, ctheta * sphi, cphi * ctheta, zeros, zeros, zeros], dim=1)
        f4 = torch.cat([zeros, zeros, zeros, ones, sphi * ttheta, cphi * ttheta], dim=1)
        #f4 = torch.cat([zeros, zeros, zeros, ones, sphi, cphi], dim=1)

        f5 = torch.cat([zeros, zeros, zeros, zeros, cphi, -sphi], dim=1)
        f6 = torch.cat([zeros, zeros, zeros, zeros, sphi / ctheta, cphi / ctheta], dim=1)
        #f6 = torch.cat([zeros, zeros, zeros, zeros, sphi, cphi], dim=1)

        f = torch.stack((f1, f2, f3, f4, f5, f6), dim=-2)

        #f = 5 * torch.eye(6).to(device=state.device)
        next_state = state.unsqueeze(-1) + f @ control.unsqueeze(-1) * self.dt
        #control.register_hook(lambda x: print(torch.mean(x)))

        return next_state.squeeze(-1)

class Quadcopter12DDynamics(nn.Module):

    def __init__(self, dt):
        super().__init__()
        self.dt = dt

    def forward(self, state, control):
        ''' unroll state '''
        g = -9.81
        m = 1
        Ix, Iy, Iz = 0.5, 0.1, 0.3
        K = 5
        x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r = torch.chunk(state, chunks=12, dim=-1)

        u1, u2, u3, u4  = torch.chunk(control, chunks=4, dim=-1)

        # Trigonometric fcns on all the angles needed for dynamics
        cphi = torch.cos(phi)
        ctheta = torch.cos(theta)
        cpsi = torch.cos(psi)

        sphi = torch.sin(phi)
        stheta = torch.sin(theta)
        spsi = torch.sin(psi)

        ttheta = torch.tan(theta)

        x_ddot = -(sphi * spsi + cpsi * cphi * stheta) * K * u1 / m
        y_ddot = - (cpsi * sphi - cphi * spsi * stheta) * K * u1 / m
        z_ddot = g - (cphi * ctheta) * K * u1 / m

        p_dot = ((Iy - Iz) * q * r + K * u2) / Ix
        q_dot = ((Iz - Ix) * p * r + K * u3) / Iy
        r_dot = ((Ix - Iy) * p * q + K * u4) / Iz

        ''' velocities'''
        psi_dot = q * sphi / ctheta + r * cphi / ctheta
        theta_dot = q * cphi - r * sphi
        phi_dot = p + q * sphi * ttheta + r * cphi * ttheta

        dstate = torch.cat((x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot,
                           x_ddot, y_ddot, z_ddot, p_dot, q_dot, r_dot), dim=-1)

        return state + dstate * self.dt



class UniformActionPrior(nn.Module):

    def __init__(self, upper, lower):
        super().__init__()
        self.register_buffer('ub', torch.tensor(upper))
        self.register_buffer('lb', torch.tensor(lower))

    def log_prob(self, actions):
        actions_above_lower = torch.where(actions > self.lb, torch.zeros_like(actions), -1 + actions - self.lb)
        actions_below_upper = torch.where(actions < self.ub, torch.zeros_like(actions), -1 + self.ub - actions)
        return actions_above_lower + actions_below_upper

class QuadcopterCollisionFcn(nn.Module):

    def __init__(self, disc_height, disc_radius, dworld):
        '''
        models quadcopter as disc
        '''
        super().__init__()

        h = disc_height
        r = disc_radius
        self.dw = dworld
        self.register_buffer('T', torch.tensor([
            [0, 0, -h/2],
            [r, 0, -h/2],
            [-r, 0, -h/2],
            [0, r, -h/2],
            [0, -r, -h/2],
            [0, 0, h/2],
            [r, 0, h/2],
            [-r, 0, h/2],
            [0, r, h/2],
            [0, -r, h/2]
        ]).transpose(0, 1))

        self.collision_fn = CollisionFcn()

    def apply(self, state, sdf, sdf_grad):
        # create rotation matrix
        # state is B x N x dx
        # Want to turn it to B x N x 3 x 3
        # orientation as euler angles
        orientation = state[:, :, 3:6]
        position = state[:, :, :3]

        B, N, _ = state.shape

        orientation = torch.flip(orientation, [-1])
        R = euler_angles_to_matrix(orientation, convention='ZYX')

        check_points = R @ self.T
        check_points = check_points.transpose(-1, -2)[:, :, :, :self.dw]
        check_points = position[:, :, :self.dw].unsqueeze(2).repeat(1, 1, 10, 1) + check_points

        collision = self.collision_fn.apply(check_points.reshape(B, -1, self.dw), sdf, sdf_grad)
        collision = torch.min(collision.reshape(B, N, -1), dim=-1).values

        #torch.where(torch.clamp(collision.sum(dim=-1), min=torch.min(collision), max=torch.max(collision))
        return torch.where(collision < 0, -1e4 * torch.ones(B, N, device=collision.device),
                           torch.zeros(B, N, device=collision.device))

class QuadcopterModel(GenerativeModel):

    def __init__(self, world_dim=2, dt=0.01, kinematic=True):
        assert world_dim == 2 or world_dim == 3
        self.dworld = world_dim
        if kinematic:
            dynamics = Quadcopter6DDynamics(dt=dt)
            state_dim = 6
            control_dim = 6
            self.dynamic = False
        else:
            dynamics = Quadcopter12DDynamics(dt=dt)
            state_dim = 12
            control_dim = 4
            self.dynamic = True

        #prior = UniformActionPrior(upper=5.0, lower=-5.0)
        prior = torch.distributions.Normal(loc=0.0, scale=1.0)
        super().__init__(dynamics=dynamics, action_prior=prior, state_dim=state_dim, control_dim=control_dim)
        self.collision_fn = QuadcopterCollisionFcn(0.05, 0.1, world_dim)

    def goal_log_likelihood(self, state, goal):
        state_config = state[:, :, :self.dworld]
        goal_config = goal[:, :, :self.dworld]
        goal_ll = -torch.norm(state_config - goal_config, dim=-1)

        if self.dworld == 2:
            goal_ll = goal_ll - state[:, :, 2].abs()

        goal_ll = goal_ll# - state[:, :, 3].abs() - state[:, :, 4].abs()
        if self.dynamic:
            velocity_penalty = torch.norm(state[:, :, 9:], dim=-1)
            # clip velocity penalty as it blows up during training
            velocity_penalty = torch.clamp(velocity_penalty, min=None, max=1e5)
            goal_ll = goal_ll - 0.1 * velocity_penalty

        #minimise orientation
        return 10 * goal_ll# - (state[:, :, 3]**2 + state[:, :, 4]**2)

    def collision_log_likelihood(self, state, sdf, sdf_grad):
        config = state[:, :, :self.dworld]
        collision_ll = self.collision_fn.apply(state, sdf, sdf_grad)
        return collision_ll
        constraint_ll = -1000 * self.orientation_constraints(state)
        return collision_ll + constraint_ll

    def orientation_constraints(self, state):
        orientation = state[:, :, 3:5]
        over_lim = torch.where(orientation.abs() > np.pi / 3.0, torch.ones_like(orientation), torch.zeros_like(orientation))
        violate_constraints = torch.sum(over_lim, dim=-1)
        return violate_constraints