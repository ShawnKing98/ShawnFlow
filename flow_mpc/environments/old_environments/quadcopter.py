import numpy as np
from flow_mpc.environments.environment import Environment
from flow_mpc.environments.halton import halton
from sdf_tools import utils_3d


class Quadcopter(Environment):
    """
    Class for quadcopter navigation in cluttered environments.
    """

    def __init__(self, state_dim, control_dim, sdf_shape):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.sdf_shape = sdf_shape
        self.grid = np.zeros(sdf_shape)

        self.env_type = 'spheres'

        # Get halton samples for randomizing environment
        self.n_halton_samples = 100000
        self.halton_samples = halton(dim=3, n_sample=self.n_halton_samples)

    def _gen_spheres_env(self):
        # randomise obstacles
        self.num_obstacles = np.random.randint(low=3, high=15)
        halton_start_idx = np.random.randint(low=0, high=self.n_halton_samples - self.num_obstacles - 1)
        halton_end_idx = halton_start_idx + self.num_obstacles

        self.obstacle_positions = 3.8 * (self.halton_samples[halton_start_idx:halton_end_idx] - 0.5)

        # self.obstacle_positions = self.obstacle_nominal_positions + np.random.uniform(-0.4, 0.4, size=(self.num_obstacles, 2))
        # bias towards small obstacles?
        min_size = 0.25
        self.obstacle_radii = min_size + 0.25 * np.random.uniform(size=self.num_obstacles)
        # save grid
        self._get_spheres_grid()

    def _get_spheres_grid(self):
        imsize = 64

        environment_limits = [-2, 2]

        distance_per_pixel = (environment_limits[1] - environment_limits[0]) / imsize

        pixel_locations = np.arange(-2 + 0.5 * distance_per_pixel, 2.01 - 0.5 * distance_per_pixel, distance_per_pixel)

        pixel_x, pixel_y, pixel_z = np.meshgrid(pixel_locations, pixel_locations, pixel_locations)
        pixel_locations = np.stack((pixel_x, pixel_y, pixel_z), axis=3).reshape(imsize, imsize, imsize, 1, 3).repeat(
            self.num_obstacles,
            axis=3)

        obstacle_centres = self.obstacle_positions.reshape(1, 1, 1, -1, 3).repeat(imsize, axis=0).repeat(imsize,
                                                                                                         axis=1).repeat(
            imsize, axis=2)
        obstacle_radii = self.obstacle_radii.reshape(1, 1, 1, -1).repeat(imsize, axis=0).repeat(imsize, axis=1).repeat(
            imsize, axis=2)

        distance_to_obstacle_centres = np.linalg.norm(pixel_locations - obstacle_centres, axis=4)
        # distance_to_obstacle_edges = np.clip(distance_to_obstacle_centres - obstacle_radii, a_min=0, a_max=None)
        distance_to_obstacle_edges = distance_to_obstacle_centres - obstacle_radii
        sdf = distance_to_obstacle_edges.min(axis=3)
        self.grid = np.where(sdf < 0, np.ones_like(sdf), np.zeros_like(sdf))

    def _gen_narrow_passages_env(self):
        # TODO generalise to different size grids? seems annoying to do
        grid = np.zeros((64, 64, 64))

        # Make 3 'walls' which split the space into 8 rooms
        wall_thickness = 6
        grid[32-wall_thickness//2:32+wall_thickness//2] = 1.0
        grid[:, 32-wall_thickness//2:32+wall_thickness//2] = 1.0
        grid[:, :, 32-wall_thickness//2:32+wall_thickness//2] = 1.0

        # put doors in x-y plabne
        grid = self._put_four_doors_in_wall(grid, wall_thickness, wall_thickness)
        # put doors in x-z plane
        grid = self._put_four_doors_in_wall(grid.transpose(0, 2, 1), wall_thickness, wall_thickness).transpose(0, 2, 1)
        # put doors in y-z plane
        grid = self._put_four_doors_in_wall(grid.transpose(1, 2, 0), wall_thickness, wall_thickness).transpose(2, 0, 1)

        self.grid = grid

    def _put_four_doors_in_wall(self, wall, door_width, door_height):
        # First door is in quadrant 0:30, 0:30
        door_x_low = np.random.randint(low=0, high=28 - door_width)
        door_y_low = np.random.randint(low=0, high=28 - door_width)
        wall[door_x_low:door_x_low + door_width, door_y_low:door_y_low + door_height] = 0.0

        # First door is in quadrant 0:30, 0:30
        door_x_low = np.random.randint(low=0, high=28 - door_width)
        door_y_low = np.random.randint(low=36, high=64 - door_height)
        wall[door_x_low:door_x_low + door_width, door_y_low:door_y_low + door_height] = 0.0

        # First door is in quadrant 0:30, 0:30
        door_x_low = np.random.randint(low=36, high=64 - door_width)
        door_y_low = np.random.randint(low=0, high=28 - door_height)
        wall[door_x_low:door_x_low + door_width, door_y_low:door_y_low + door_height] = 0.0

        # First door is in quadrant 0:30, 0:30
        door_x_low = np.random.randint(low=36, high=64 - door_width)
        door_y_low = np.random.randint(low=36, high=64 - door_height)
        wall[door_x_low:door_x_low + door_width, door_y_low:door_y_low + door_height] = 0.0
        return wall

    def get_environment_sdf(self):
        grid_size = 64
        resolution = 4.0 / grid_size
        sdf_origin = [0.0, 0.0, 0.0]

        # occupancy_grid = np.where(self.grid > 0, 0, 1).astype(np.uint8)
        occupancy_grid = self.grid.astype(np.uint8)
        sdf, sdf_gradient = utils_3d.compute_sdf_and_gradient(occupancy_grid, resolution, sdf_origin)
        sdf[np.where(sdf < 0)] *= 1000.0
        sdf_gradient[np.where(sdf < 0)] *= 1000.0

        grid_from_sdf = np.where(sdf < 0, np.ones_like(sdf), np.zeros_like(sdf))

        return sdf, sdf_gradient

    def _convert_position_to_pixels(self, position):
        pixels = []
        for i, p in enumerate(position):
            pixels.append(int(self.sdf_shape[i] * (p + 2) / 4))
        return pixels

    def check_collision(self):
        pixels = self._convert_position_to_pixels(self.state[:3])
        if any(pixel < 0 for pixel in pixels) or any(pixel > 63 for pixel in pixels):
            return True

        return self.grid[pixels[0], pixels[1], pixels[2]]

    def render(self):
        start_voxels = np.zeros(self.sdf_shape, dtype=np.bool)
        goal_voxels = np.zeros(self.sdf_shape, dtype=np.bool)

        start_p = self._convert_position_to_pixels(self.start[:3])
        goal_p = self._convert_position_to_pixels(self.goal[:3])
        start_voxels[start_p[0], start_p[1], start_p[2]] = True
        goal_voxels[goal_p[0], goal_p[1], goal_p[2]] = True

        colours = np.empty(self.sdf_shape, dtype=object)
        colours[start_voxels] = 'red'
        colours[goal_voxels] = 'blue'
        colours[self.grid.astype(dtype=np.bool)] = 'green'

        voxels = np.where(self.grid + start_voxels + goal_voxels < 1, np.zeros_like(self.grid), np.ones_like(self.grid))
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d
        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(voxels, facecolors=colours, edgecolor='k')

        plt.show()


class Quadcopter6DFullyActuated(Quadcopter):
    """
    Class for quadcopter navigation in cluttered environments.
    """

    def __init__(self, dt):
        super().__init__(state_dim=6, control_dim=6, sdf_shape=(64, 64, 64))
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
                [0, 0, 0, 0, sphi * ctheta, cphi / ctheta]
            ]
        )

        self.state += f1 @ control * self.dt
        return self.state, self.check_collision()

    def reset(self):
        if self.env_type == 'spheres':
            self._gen_spheres_env()
        elif self.env_type == 'narrow_passages':
            self._gen_narrow_passages_env()
        else:
            raise ValueError('Invalid env type')

        self.sdf, self.sdf_grad = self.get_environment_sdf()

        self.reset_start_and_goal()

    def reset_start_and_goal(self):
        start_and_goal_in_collision = True
        while start_and_goal_in_collision:
            # reset start and goal
            start = np.zeros(6)
            # positions
            start[:3] = 1.8 - 3.6 * np.random.rand(3)
            # Angles in rad
            start[3:] = -np.pi + 2 * np.pi * np.random.rand(3)

            start_pixels = self._convert_position_to_pixels(start[:3])
            start_distance_to_ob = self.sdf[start_pixels[0], start_pixels[1], start_pixels[2]]
            if start_distance_to_ob < 0.2:
                continue

            goal = np.zeros(6)
            # positions
            goal[:3] = 1.8 - 3.6 * np.random.rand(3)
            # Angles in rad
            goal[3:] = -np.pi + 2 * np.pi * np.random.rand(3)

            goal_pixels = self._convert_position_to_pixels(goal[:3])
            goal_distance_to_ob = self.sdf[goal_pixels[0], goal_pixels[1], goal_pixels[2]]

            if goal_distance_to_ob < 0.2:
                continue

            min_goal_distance = 4 * np.random.rand()
            if np.linalg.norm(goal[:3] - start[:3]) < min_goal_distance:
                continue

            start_and_goal_in_collision = False

        self.start = start
        self.goal = goal
        self.state = self.start.copy()


if __name__ == '__main__':
    env = Quadcopter6DFullyActuated(dt=0.01)
    env.reset()
    env.render()
