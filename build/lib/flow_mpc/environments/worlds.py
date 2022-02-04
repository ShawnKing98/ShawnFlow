import numpy as np

from flow_mpc.environments.halton import halton
from sdf_tools import utils_2d, utils_3d

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class World:

    def __init__(self, dim_world, world_size=4):
        if dim_world not in [2, 3]:
            raise ValueError('World must be either 2D or 3D')
        self.dw = dim_world
        self.world_size = world_size
        self.n_halton_samples = 100000
        self.halton_samples = halton(dim=dim_world, n_sample=self.n_halton_samples)

        self.grid_size = tuple([64] * self.dw)
        self.grid = np.zeros(self.grid_size)
        self.sdf = np.zeros(self.grid_size)
        self.sdf_grad = np.zeros(self.grid_size)

    def reset(self):
        self.grid = self._get_occupancy_grid()
        self.sdf, self.sdf_grad = self.get_environment_sdf()

    def _get_occupancy_grid(self):
        raise NotImplementedError

    def get_environment_sdf(self):
        grid_size = 64
        resolution = self.world_size / grid_size
        sdf_origin = [0] * self.dw

        # occupancy_grid = np.where(self.grid > 0, 0, 1).astype(np.uint8)
        occupancy_grid = self.grid.astype(np.uint8)
        if self.dw == 2:
            utils = utils_2d
        elif self.dw == 3:
            utils = utils_3d

        sdf, sdf_gradient = utils.compute_sdf_and_gradient(occupancy_grid, resolution, sdf_origin)
        sdf[np.where(sdf < 0)] *= 1000.0
        sdf_gradient[np.where(sdf < 0)] *= 1000.0


        return sdf, sdf_gradient

    def render(self):
        if self.dw == 2:
            plt.imshow(self.grid)
            plt.show()
        else:
            ax = plt.figure().add_subplot(projection='3d')
            ax.voxels(self.grid, facecolors='k', edgecolor='k')
            plt.show()

    def position_to_pixels(self, position):
        assert len(position) == self.dw
        half_size = self.world_size / 2
        pixels = (np.array(self.grid_size) * (np.array(position) + half_size) / self.world_size).astype(np.int16)
        # For 2D we reverse the pixels as it the grid is [y, x]
        if self.dw == 2:
            return pixels[::-1]
        return pixels

    def distance_to_pixels(self, distance):
        half_size = self.world_size / 2
        pixels = (np.array(self.grid_size) * (np.array(distance)) / self.world_size).astype(np.int16)
        return pixels

    def check_bounds(self, position):
        if np.max(position) >= self.grid_size[0] or np.min(position) < 0:
            return True
        return False

    def check_collision(self, position):
        pixels = self.position_to_pixels(position)
        if not self.check_bounds(pixels):
            return self.grid[tuple(pixels)]
        return True


class SphereWorld(World):

    def __init__(self, world_dim, world_size=4, min_obstacles=3, max_obstacles=9, min_radius=0.35, max_radius=0.6):
        super().__init__(dim_world=world_dim, world_size=world_size)
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self.max_r = max_radius
        self.min_r = min_radius
        self.obstacle_distance_norm = 2

    def _get_occupancy_grid(self):
        obstacle_positions, obstacle_radii = self._generate_environment()
        occupancy_grid = self._get_occupancy(obstacle_positions, obstacle_radii)
        return occupancy_grid

    def _generate_environment(self):
        # randomise number of obstacles
        num_obstacles = np.random.randint(low=self.min_obstacles, high=self.max_obstacles)

        # Sample object positions from halton sequence
        halton_start_idx = np.random.randint(low=0, high=self.n_halton_samples - num_obstacles - 1)
        halton_end_idx = halton_start_idx + num_obstacles

        # obstacle positions in range [-0.5, 0.5]
        obstacle_positions = self.halton_samples[halton_start_idx:halton_end_idx] - 0.5
        obstacle_positions *= 0.95 * self.world_size

        # Randomise obsacle radii
        obstacle_radii = self.min_r + (self.max_r - self.min_r) * np.random.uniform(size=num_obstacles)

        return obstacle_positions, obstacle_radii

    def _get_occupancy(self, obstacle_positions, obstacle_radii):
        imsize = 64
        N_obstacles = obstacle_positions.shape[0]

        environment_limits = [-self.world_size / 2, self.world_size / 2]
        distance_per_pixel = (environment_limits[1] - environment_limits[0]) / imsize

        # Generate all pixel locations
        pixel_locations = np.arange(-self.world_size/2 + 0.5 * distance_per_pixel,
                                    self.world_size/2 + .01 - 0.5 * distance_per_pixel,
                                    distance_per_pixel)

        pixel_locations = np.meshgrid(*([pixel_locations] * self.dw))
        pixel_locations = np.expand_dims(np.stack(pixel_locations, axis=-1), axis=-2).repeat(N_obstacles, axis=-2)

        for _ in range(self.dw):
            obstacle_positions = np.expand_dims(obstacle_positions, axis=0).repeat(imsize, axis=0)
            obstacle_radii = np.expand_dims(obstacle_radii, axis=0).repeat(imsize, axis=0)

        distance_to_obstacle_centres = np.linalg.norm(pixel_locations - obstacle_positions,
                                                      axis=-1, ord=self.obstacle_distance_norm)
        distance_to_obstacle_edges = np.clip(distance_to_obstacle_centres - obstacle_radii, a_min=0, a_max=None)
        if N_obstacles == 0:
            grid = np.zeros(distance_to_obstacle_edges.shape[0:2])
        else:
            sdf = distance_to_obstacle_edges.min(axis=-1)
            grid = 1.0 - np.where(sdf > 0, np.ones_like(sdf), np.zeros_like(sdf))

        # add borders
        # self.add_rectangle(self.grid, (-2, -2), width=0.2, height=4)
        # self.add_rectangle(self.grid, (1.9, -2), width=0.2, height=4)
        # self.add_rectangle(self.grid, (-2, -2), width=4, height=0.2)
        # self.add_rectangle(self.grid, (-1.9, 1.9), width=4, height=0.2)
        # self.grid = 1.0 - self.grid
        return grid


class SquareWorld(SphereWorld):
    def __init__(self, world_dim, world_size=4, min_obstacles=1, max_obstacles=14, min_width=0.2, max_width=0.4):
        super().__init__(world_dim, world_size, min_obstacles, max_obstacles, min_width, max_width)
        self.obstacle_distance_norm = 1


class NarrowPassagesWorld(World):
    def __init__(self, world_dim, world_size=4, wall_thickness=6):
        super().__init__(dim_world=world_dim, world_size=world_size)
        self.wall_thickness = wall_thickness
        self.door_width = 8

    def _get_occupancy_grid(self):
        # TODO generalise to different size grids? seems annoying to do
        imsize = 64

        grid_size = tuple([imsize] * self.dw)
        grid = np.zeros(grid_size)

        # Make 3 'walls' which split the space into 8 rooms
        wall_thickness = self.wall_thickness

        # Put wall along x direction
        grid[32 - wall_thickness // 2:32 + wall_thickness // 2] = 1.0
        # Put wall along y direction
        grid[:, 32 - wall_thickness // 2:32 + wall_thickness // 2] = 1.0

        if self.dw == 3:
            # Put wall along z direction
            grid[:, :, 32 - wall_thickness // 2:32 + wall_thickness // 2] = 1.0

        if self.dw == 2:
            grid = self._put_two_doors_in_wall(grid, self.door_width)
            grid = self._put_two_doors_in_wall(grid.transpose(1, 0), self.door_width).transpose(1, 0)

        elif self.dw == 3:
            # put doors in x-y plane
            grid = self._put_four_doors_in_wall(grid, self.door_width, self.door_width)
            # put doors in x-z plane
            grid = self._put_four_doors_in_wall(grid.transpose(0, 2, 1), self.door_width, self.door_width).transpose(0,
                                                                                                                     2,
                                                                                                                     1)
            # put doors in y-z plane
            grid = self._put_four_doors_in_wall(grid.transpose(1, 2, 0), self.door_width, self.door_width).transpose(2,
                                                                                                                     0,
                                                                                                                     1)

        return grid

    def _put_two_doors_in_wall(self, wall, door_width):
        # First door is in half 0:30
        door_low = np.random.randint(low=8, high=28 - door_width)
        wall[door_low:door_low + door_width] = 0.0

        # First door is in half 34:64
        door_low = np.random.randint(low=36, high=56 - door_width)
        wall[door_low:door_low + door_width] = 0.0
        return wall

    def _put_four_doors_in_wall(self, wall, door_width, door_height):
        # First door is in quadrant 0:30, 0:30
        door_x_low = np.random.randint(low=8, high=28 - door_width)
        door_y_low = np.random.randint(low=8, high=28 - door_width)
        wall[door_x_low:door_x_low + door_width, door_y_low:door_y_low + door_height] = 0.0

        # First door is in quadrant 0:32, 32:64
        door_x_low = np.random.randint(low=8, high=28 - door_width)
        door_y_low = np.random.randint(low=36, high=56 - door_height)
        wall[door_x_low:door_x_low + door_width, door_y_low:door_y_low + door_height] = 0.0

        # First door is in quadrant 32:64, 0:32
        door_x_low = np.random.randint(low=36, high=56 - door_width)
        door_y_low = np.random.randint(low=8, high=28 - door_height)
        wall[door_x_low:door_x_low + door_width, door_y_low:door_y_low + door_height] = 0.0

        # First door is in quadrant 32:64, 32:64
        door_x_low = np.random.randint(low=36, high=56 - door_width)
        door_y_low = np.random.randint(low=36, high=56 - door_height)
        wall[door_x_low:door_x_low + door_width, door_y_low:door_y_low + door_height] = 0.0
        return wall


if __name__ == '__main__':
    # world = NarrowPassagesWorld(world_dim=3, world_size=4)
    world = SphereWorld(world_dim=2, world_size=4)
    # world.reset()
    # world.render()

    # world = SquareWorld(world_dim=2, world_size=4)
    # world.reset()
    # world.render()

    # world = NarrowPassagesWorld(world_dim=3, world_size=4, wall_thickness=2)
    world.reset()
    print(world._convert_position_to_pixels(np.ones(2)))
    for _ in range(10):
        print(world.check_collision(np.random.uniform(low=-2, high=2, size=2)))
    world.render()
