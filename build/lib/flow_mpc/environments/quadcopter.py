import numpy as np
from flow_mpc.environments.environment import Environment


class QuadcopterEnv(Environment):
    """
    Class for quadcopter navigation in cluttered environments.
    """

    def __init__(self, world_dim, world_type='spheres', dt=0.025):
        super().__init__(6, 6, world_dim, world_type)
        self.dt = dt

    def step(self, control):
        # Unroll state
        x, y, z, phi, theta, psi = self.state

        # Trigonometric fcns on all the angles needed for dynamics
        cphi = np.cos(phi)
        ctheta = np.cos(theta)
        cpsi = np.cos(psi)
        sphi = np.sin(phi)
        stheta = np.sin(theta)
        spsi = np.sin(psi)
        ttheta = np.tan(theta)

        f1 = np.array(
            [
                [ctheta * cpsi, -cphi * spsi + cpsi * sphi * stheta, spsi * sphi + cphi * cpsi * stheta, 0, 0, 0],
                [ctheta * spsi, cphi * cpsi + sphi * spsi * stheta, -cpsi * sphi + cphi * spsi * stheta, 0, 0, 0],
                [-stheta, ctheta * sphi, cphi * ctheta, 0, 0, 0],
                [0, 0, 0, 1, sphi * ttheta, cphi * ttheta],
                [0, 0, 0, 0, cphi, -sphi],
                [0, 0, 0, 0, sphi / ctheta, cphi / ctheta]
            ]
        )
        self.state = self.state + f1 @ control * self.dt
        return self.state, self.world.check_collision(self.state[:self.world.dw])

    def reset_start_and_goal(self):
        start_and_goal_in_collision = True
        while start_and_goal_in_collision:
            # reset start and goal
            start = np.zeros(self.state_dim)
            goal = np.zeros(self.state_dim)

            # positions
            start[:3] = self.world.world_size * (0.45 - 0.9 * np.random.rand(3))
            goal[:3] = self.world.world_size * (0.45 - 0.9 * np.random.rand(3))

            # Angles in rad
            # Heading angle can be anything
            start[5] = -np.pi + 2 * np.pi * np.random.rand()
            goal[5] = -np.pi + 2 * np.pi * np.random.rand()
            # Other two must be restriced by a lot
            start[3:5] = 2 * np.random.rand(2) - 1  # 0.1*(-np.pi + 2 * np.pi * np.random.rand(2))
            goal[3:5] = 2 * np.random.rand(2) - 1  # 0.1*(-np.pi + 2 * np.pi * np.random.rand(2))

            # If we are in 2D world we only care about obstacles in x y plane
            start_pixels = self.world.position_to_pixels(start[:self.world.dw])
            goal_pixels = self.world.position_to_pixels(goal[:self.world.dw])

            start_distance_to_ob = self.world.sdf[tuple(start_pixels)]
            goal_distance_to_ob = self.world.sdf[tuple(goal_pixels)]

            if start_distance_to_ob < 0.2 or goal_distance_to_ob < 0.2:
                continue

            min_goal_distance = 4 * np.random.rand()
            if np.linalg.norm(goal[:3] - start[:3]) < min_goal_distance:
                continue

            start_and_goal_in_collision = False

        self.start = start
        self.goal = goal
        self.state = self.start.copy()

        return True


    def cost(self):
        return np.linalg.norm(self.state[:self.world.dw] - self.goal[:self.world.dw])

    def __str__(self):
        return f'quadcopter_{self.world_type}_{self.world.dw}D'

    def at_goal(self):
        return self.cost() < 0.1 * self.world.dw


class QuadcopterDynamicEnv(Environment):
    """
    Class for dynamic quadcopter navigation in cluttered environments.
    """

    def __init__(self, world_dim, world_type='spheres', dt=0.025):
        super().__init__(12, 4, world_dim, world_type)
        self.dt = dt

    def step(self, control):
        # Unroll state
        g = -9.81
        m = 1
        Ix, Iy, Iz = 0.5, 0.1, 0.3
        K = 5
        x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r = self.state

        u1, u2, u3, u4 = control

        # Trigonometric fcns on all the angles needed for dynamics

        cphi = np.cos(phi)
        ctheta = np.cos(theta)
        cpsi = np.cos(psi)
        sphi = np.sin(phi)
        stheta = np.sin(theta)
        spsi = np.sin(psi)
        ttheta = np.tan(theta)

        ''' accelerations first '''
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

        dstate = np.stack((x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot,
                           x_ddot, y_ddot, z_ddot, p_dot, q_dot, r_dot), axis=-1)

        self.state = self.state + dstate * self.dt
        # self.state[3:6] = normalize_angles(self.state[3:6])

        return self.state, self.world.check_collision(self.state[:self.world.dw])

    def reset_start_and_goal(self):
        start_and_goal_in_collision = True
        while start_and_goal_in_collision:
            # reset start and goal
            start = np.zeros(self.state_dim)
            goal = np.zeros(self.state_dim)

            # positions
            start[:3] = self.world.world_size * (0.45 - 0.9 * np.random.rand(3))
            goal[:3] = self.world.world_size * (0.45 - 0.9 * np.random.rand(3))

            # Angles in rad
            # Heading angle can be anything
            start[5] = -np.pi + 2 * np.pi * np.random.rand()
            goal[5] = -np.pi + 2 * np.pi * np.random.rand()
            # Other two must be restriced by a lot
            start[3:5] = 2 * np.random.rand(2) - 1  # 0.1*(-np.pi + 2 * np.pi * np.random.rand(2))
            goal[3:5] = 2 * np.random.rand(2) - 1  # 0.1*(-np.pi + 2 * np.pi * np.random.rand(2))

            ''' velocities sampled from random normal -- make angular velocities smaller '''
            start[6:] = np.random.randn(6)
            start[9:] *= 5
            ''' goal is always zero velocity'''

            # If we are in 2D world we only care about obstacles in x y plane
            start_pixels = self.world.position_to_pixels(start[:self.world.dw])
            goal_pixels = self.world.position_to_pixels(goal[:self.world.dw])

            start_distance_to_ob = self.world.sdf[tuple(start_pixels)]
            goal_distance_to_ob = self.world.sdf[tuple(goal_pixels)]

            if start_distance_to_ob < 0.2 or goal_distance_to_ob < 0.2:
                continue

            min_goal_distance = 4 * np.random.rand()
            if np.linalg.norm(goal[:3] - start[:3]) < min_goal_distance:
                continue

            start_and_goal_in_collision = False

        self.start = start
        self.goal = goal
        self.state = self.start.copy()

        return True

    def cost(self):
        return np.linalg.norm(self.state[:self.world.dw] - self.goal[:self.world.dw])

    def __str__(self):
        return f'quadcopter_{self.world_type}_{self.world.dw}D'

    def at_goal(self):
        return self.cost() < 0.1 * self.world.dw


def normalize_angles(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


if __name__ == '__main__':
    env = QuadcopterEnv(dt=0.01, world_dim=3)
    env.reset()
    env.world.render()
