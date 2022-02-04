import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flow_mpc.environments.environment import Environment


class DoubleIntegratorEnv(Environment):

    def __init__(self, world_dim=2, world_type='spheres', dt=0.05, action_noise_cov=None):
        # Double integrator in either 2D or 3D
        state_dim = 2 * world_dim
        control_dim = world_dim
        super().__init__(state_dim, control_dim, world_dim, world_type)

        self.dt = dt
        self.action_noise_cov = action_noise_cov
        # state is [x, y, x_dot, y_dot]
        # control is x_ddot, y_ddot
        if world_dim == 2:
            self.A = np.array([[1.0, 0.0, self.dt, 0.0],
                               [0.0, 1.0, 0.0, self.dt],
                               [0.0, 0.0, 0.95, 0.0],
                               [0.0, 0.0, 0.0, 0.95]])

            self.B = np.array([[0.0, 0.0],
                               [0.0, 0.0],
                               [self.dt, 0.0],
                               [0.0, self.dt]])
        elif world_dim == 3:
            self.A = np.array([[1.0, 0.0, 0.0, self.dt, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0, self.dt, 0.0],
                               [0.0, 0.0, 1.0, 0.0, 0.0, self.dt],
                               [0.0, 0.0, 0.0, 0.95, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.95, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.95]])

            self.B = np.array([[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [self.dt, 0.0, 0.0],
                               [0.0, self.dt, 0.0],
                               [0.0, 0.0, self.dt]
                               ])

    def reset_start_and_goal(self):
        start_goal_in_collision = True

        while start_goal_in_collision:
            self.start = np.zeros(self.state_dim)
            # randomise position
            self.start[:self.state_dim // 2] = self.world.world_size * (
                    0.45 - 0.9 * np.random.rand(self.state_dim // 2))
            # randmise velocity
            self.start[self.state_dim // 2:] = 0.25 * np.random.randn(self.state_dim // 2)
            self.state = self.start.copy()

            # goal
            self.goal = np.zeros(self.state_dim)
            # randomise position
            self.goal[:self.state_dim // 2] = self.world.world_size * (
                    0.45 - 0.9 * np.random.rand(self.state_dim // 2))

            # check start in collision
            start_pixels = self.world.position_to_pixels(self.start[:self.state_dim // 2])
            goal_pixels = self.world.position_to_pixels(self.goal[:self.state_dim // 2])

            start_distance_to_obs = self.world.sdf[tuple(start_pixels)]
            goal_distance_to_obs = self.world.sdf[tuple(goal_pixels)]

            start_goal_in_collision = (goal_distance_to_obs < 0.2) or (start_distance_to_obs < 0.2)
            min_goal_distance = 4 * np.random.rand()
            if np.linalg.norm(self.goal[:2] - self.start[:2]) < min_goal_distance:
                start_goal_in_collision = True

        return True

    def render(self, add_start_and_goal=True, trajectory=None):
        self.world.render()

    def step(self, control):
        if self.action_noise_cov is not None:
            control = control + np.random.multivariate_normal(np.zeros(self.control_dim), self.action_noise_cov)
        self.state = self.A @ self.state + self.B @ control
        position = self.state[:self.state_dim // 2]
        return self.state, self.world.check_collision(position)

    def cost(self):
        return np.linalg.norm(self.state[:self.world.dw] - self.goal[:self.world.dw])

    def __str__(self):
        return f'double_integrator_{self.world_type}_{self.world.dw}D'


if __name__ == '__main__':
    env = DoubleIntegratorEnv(world_dim=3, world_type='spheres')
    env.reset()
    print(env.start)
    print(env.goal)
    print(env.world.position_to_pixels(env.start[:3]))
    print(env.world.position_to_pixels(env.goal[:3]))

    print(env.step(np.ones(3)))
