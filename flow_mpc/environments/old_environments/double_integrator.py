import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flow_mpc.environments.halton import halton
from sdf_tools import utils_2d
import cv2

class DoubleIntegrator2DEnv:

    def __init__(self):
        self.dt = 0.05
        # state is [x, y, x_dot, y_dot]
        # control is x_ddot, y_ddot
        self.A = np.array([[1.0, 0.0, self.dt, 0.0],
                           [0.0, 1.0, 0.0, self.dt],
                           [0.0, 0.0, 0.95, 0.0],
                           [0.0, 0.0, 0.0, 0.95]])

        self.B = np.array([[0.0, 0.0],
                           [0.0, 0.0],
                           [self.dt, 0.0],
                           [0.0, self.dt]])

        # can just be canonical start and goal
        self.state = np.zeros(4)
        self.start = np.zeros(4)
        self.goal = np.array([1.0, 1.0, 0.0, 0.0])

        # canonical obstacles (let's say 3 circular obstacles)
        self.num_obstacles = 0
        self.obstacle_positions = np.zeros((self.num_obstacles, 2))
        self.obstacle_radii = np.zeros((self.num_obstacles))
        self.sdf_size = (64, 64)

        self.env_type = 'narrow_passages'
        self.env_type = 'spheres'
        self.grid = np.zeros(self.sdf_size)
        self.min_goal_distance = 3.5
        self.n_halton_samples = 1000
        self.halton_samples = halton(dim=2, n_sample=self.n_halton_samples)

    def reset(self):

        if self.env_type == 'spheres':
            self.gen_sphere_env()

        elif self.env_type == 'narrow_passages':
            self.gen_narrow_passage_env()

        elif self.env_type == 'rectangles':
            self.gen_rect_env()

        self.sdf, self.sdf_grad = self.get_environment_sdf()
        self.reset_start_and_goal()

    def gen_sphere_env(self):
        # randomise obstacles
        self.num_obstacles = np.random.randint(low=3, high=14)
        halton_start_idx = np.random.randint(low=0, high=self.n_halton_samples-self.num_obstacles-1)
        halton_end_idx = halton_start_idx + self.num_obstacles

        self.obstacle_positions = 3.8 * (self.halton_samples[halton_start_idx:halton_end_idx]- 0.5)

        #self.obstacle_positions = self.obstacle_nominal_positions + np.random.uniform(-0.4, 0.4, size=(self.num_obstacles, 2))
        # bias towards small obstacles?
        min_size = 0.2
        max_size = min(0.05 * (14 - self.num_obstacles), 0.21)
        self.obstacle_radii = min_size + 0.3 * np.random.uniform(size=self.num_obstacles)
        # save grid
        self._get_spheres_grid()

    def gen_rect_env(self):
        # randomise obstacles
        self.num_obstacles = np.random.randint(low=3, high=14)
        halton_start_idx = np.random.randint(low=0, high=self.n_halton_samples - self.num_obstacles - 1)
        halton_end_idx = halton_start_idx + self.num_obstacles

        self.obstacle_positions = 3.8 * (self.halton_samples[halton_start_idx:halton_end_idx] - 0.5)

        # self.obstacle_positions = self.obstacle_nominal_positions + np.random.uniform(-0.4, 0.4, size=(self.num_obstacles, 2))
        # bias towards small obstacles?
        min_size = 0.2
        max_size = min(0.05 * (14 - self.num_obstacles), 0.21)
        self.obstacle_dims = min_size + 0.3 * np.random.uniform(size=(self.num_obstacles, 2))
        # save grid
        self._get_rect_grid()

    def _convert_position_to_pixels(self, position):
        return int(self.sdf_size[0] * (position[0] + 2) / 4), int(self.sdf_size[1] * (position[1] + 2) / 4)

    def _convert_distance_to_pixels(self, distance):
        return int(self.sdf_size[0] * (distance / 4))

    def add_rectangle(self, image, upper_left_corner, width, height):
        upper_left = self._convert_position_to_pixels(upper_left_corner)
        w = self._convert_distance_to_pixels(width)
        h = self._convert_distance_to_pixels(height)
        image[max(0, upper_left[1]):min(self.sdf_size[0], upper_left[1] + h),
              max(0, upper_left[0]):min(self.sdf_size[1], upper_left[0] + w)] = 1.0

    def gen_narrow_passage_env(self):
        # narrow passage will be two rectangles with a narrow gap between them
        im = np.zeros(self.sdf_size).astype(np.uint8)
        wall_thickness = 0.4
        # add borders
        self.add_rectangle(im, (-2, -2), width=0.2, height=4)
        self.add_rectangle(im, (1.9, -2), width=0.2, height=4)
        self.add_rectangle(im, (-2, -2), width=4, height=0.2)
        self.add_rectangle(im, (-1.9, 1.9), width=4, height=0.2)

        # add three rectangles so that there are two passages
        y_pos = -0.5 + np.random.rand()
        gap_size = 0.2
        w1, w2 = 0.75 + np.random.rand(2)
        self.add_rectangle(im, (-2, y_pos), width=w1, height=wall_thickness)
        self.add_rectangle(im, (-2 + w1 + gap_size, y_pos), width=w2, height=wall_thickness)
        self.add_rectangle(im, (-2 + w1 + w2 + 2 * gap_size, y_pos), width=2.0, height=wall_thickness)

        # Add three rectangle vertically to allow two passages
        x_pos = -2 + w1 + gap_size + 0.5 * np.random.rand()
        h1 = 1.75 + y_pos - 0.5 * np.random.rand()
        h2 = y_pos + wall_thickness - (-2 + h1 + gap_size) + 0.5 * np.random.rand()
        #h2 = 0.75 + 0.5 * np.random.rand()
        self.add_rectangle(im, (x_pos, -2), width=wall_thickness, height=h1)
        self.add_rectangle(im, (x_pos, -2 + h1 + gap_size), width=wall_thickness, height=h2)
        self.add_rectangle(im, (x_pos, -2 + h1 + h2 + 2 * gap_size), width=wall_thickness, height=2)

        self.grid = 1 - im

    def reset_start_and_goal(self):
        start_goal_in_collision = True

        while start_goal_in_collision:
            self.start = np.zeros(4)
            self.start[0] = 1.8 - 3.6 * np.random.rand()
            self.start[1] = 1.8 - 3.6 * np.random.rand()
            self.start[2:] = 0.25 * np.random.randn(2)
            self.state = self.start.copy()

            # check start in collision
            start_pixels = self._convert_position_to_pixels(self.start[:2])
            #start_in_collision = 1 - self.grid[start_pixels[1], start_pixels[0]]

            start_distance_to_obs = self.sdf[start_pixels[1], start_pixels[0]]

            # goal
            self.goal = np.zeros(4)
            self.goal[0] = 1.8 - 3.6 * np.random.rand()
            self.goal[1] = 1.8 - 3.6 * np.random.rand()

            goal_pixels = self._convert_position_to_pixels(self.goal[:2])
            #goal_in_collision = 1 - self.grid[goal_pixels[1], goal_pixels[0]]
            goal_distance_to_obs = self.sdf[goal_pixels[1], goal_pixels[0]]

            #start_goal_in_collision = goal_in_collision or start_in_collision
            start_goal_in_collision = (goal_distance_to_obs < 0.2) or (start_distance_to_obs < 0.2)
            min_goal_distance = 4 * np.random.rand()
            min_goal_distance = 2
            if np.linalg.norm(self.goal[:2] - self.start[:2]) < min_goal_distance:
                start_goal_in_collision = True

    def check_point_in_obstacles(self, point):
        for position, radius in zip(self.obstacle_positions, self.obstacle_radii):
            if ((point[0] - position[0]) ** 2 + (point[1] - position[1]) ** 2) < (0.05 + radius) ** 2:
                return True

        return False

    def render(self, add_start_and_goal=True, trajectory=None):
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)

        if add_start_and_goal:
            goal = plt.Circle(tuple(self.goal[:2]), 0.05, color='r')
            start = plt.Circle(tuple(self.start[:2]), 0.05, color='b')
            final = plt.Circle(tuple(self.state[:2]), 0.05, color='g')

            ax.add_artist(goal)
            ax.add_artist(start)
            ax.add_artist(final)

        obstacles = []
        for position, radius in zip(self.obstacle_positions, self.obstacle_radii):
            obstacles.append(plt.Circle(tuple(position), radius, color='k', alpha=1))
            ax.add_artist(obstacles[-1])

        if trajectory is not None:
            ax.plot(trajectory[:, 0], trajectory[:, 1])

        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])

        ax.axis('off')
        ax.margins(0)
        fig.tight_layout(pad=0)
        #plt.plot()
        #plt.show()
        canvas.draw()

        s, (width, height) = canvas.print_to_buffer()
        image = np.frombuffer(s, dtype=np.uint8).reshape((height, width, -1))
        plt.close(fig)

        return image
        #return cv2.resize(image[:, :, 0], (32, 32))

    def _get_spheres_grid(self):
        imsize = 64

        environment_limits = [-2, 2]

        distance_per_pixel = (environment_limits[1] - environment_limits[0]) / imsize

        pixel_locations = np.arange(-2 + 0.5 * distance_per_pixel, 2.01 - 0.5 * distance_per_pixel, distance_per_pixel)

        pixel_x, pixel_y = np.meshgrid(pixel_locations, pixel_locations)
        pixel_locations = np.stack((pixel_x, pixel_y), axis=2).reshape(imsize, imsize, 1, 2).repeat(self.num_obstacles, axis=2)

        obstacle_centres = self.obstacle_positions.reshape(1, 1, -1, 2).repeat(imsize, axis=0).repeat(imsize, axis=1)
        obstacle_radii = self.obstacle_radii.reshape(1, 1, -1).repeat(imsize, axis=0).repeat(imsize, axis=1)

        distance_to_obstacle_centres = np.linalg.norm(pixel_locations - obstacle_centres, axis=3)
        distance_to_obstacle_edges = np.clip(distance_to_obstacle_centres - obstacle_radii, a_min=0, a_max=None)

        sdf = distance_to_obstacle_edges.min(axis=2)
        self.grid = 1.0 - np.where(sdf > 0, np.ones_like(sdf), np.zeros_like(sdf))

        # add borders
        self.add_rectangle(self.grid, (-2, -2), width=0.2, height=4)
        self.add_rectangle(self.grid, (1.9, -2), width=0.2, height=4)
        self.add_rectangle(self.grid, (-2, -2), width=4, height=0.2)
        self.add_rectangle(self.grid, (-1.9, 1.9), width=4, height=0.2)
        self.grid = 1.0 - self.grid

    def _get_rect_grid(self):
        imsize = 64

        environment_limits = [-2, 2]

        distance_per_pixel = (environment_limits[1] - environment_limits[0]) / imsize

        pixel_locations = np.arange(-2 + 0.5 * distance_per_pixel, 2.01 - 0.5 * distance_per_pixel, distance_per_pixel)

        pixel_x, pixel_y = np.meshgrid(pixel_locations, pixel_locations)
        pixel_locations = np.stack((pixel_x, pixel_y), axis=2).reshape(imsize, imsize, 1, 2).repeat(self.num_obstacles,
                                                                                                    axis=2)

        obstacle_centres = self.obstacle_positions.reshape(1, 1, -1, 2).repeat(imsize, axis=0).repeat(imsize, axis=1)
        obstacle_dims = self.obstacle_dims.reshape(1, 1, -1, 2).repeat(imsize, axis=0).repeat(imsize, axis=1)
        xy_distance_to_obstacle_centres = np.abs(pixel_locations - obstacle_centres)

        distance_to_obstacles = np.clip(xy_distance_to_obstacle_centres - obstacle_dims, a_min=0.0, a_max=None)

        in_obstacle_collision = np.where(np.sum(distance_to_obstacles, axis=3) > 0,
                                         np.zeros((imsize, imsize, self.num_obstacles)),
                                         np.ones((imsize, imsize, self.num_obstacles)))

        self.grid = np.clip(np.sum(in_obstacle_collision, axis=2), a_min=0, a_max=1)


        self.add_rectangle(self.grid, (-2, -2), width=0.2, height=4)
        self.add_rectangle(self.grid, (1.9, -2), width=0.2, height=4)
        self.add_rectangle(self.grid, (-2, -2), width=4, height=0.2)
        self.add_rectangle(self.grid, (-1.9, 1.9), width=4, height=0.2)
        self.grid = 1.0 - self.grid

    def get_environment_sdf(self):
        grid_size = 64
        resolution = 4.0 / grid_size
        sdf_origin = [0, 0]

        occupancy_grid = np.where(self.grid > 0, 0, 1).astype(np.uint8)

        print(type(occupancy_grid), type(resolution), type(sdf_origin))
        sdf, sdf_gradient = utils_2d.compute_sdf_and_gradient(occupancy_grid, resolution, sdf_origin)
        sdf_raw = sdf.copy()
        sdf[np.where(sdf < 0)] *= 1000.0
        sdf_gradient[np.where(sdf < 0)] *= 1000.0

        print(sdf)
        if True:
            fig, axes= plt.subplots(1, 3)
            axes[0].imshow(sdf_raw)
            axes[1].imshow(sdf)
            #axes[0].imshow(self.grid)
            #axes[1].imshow(sdf)
            axes[2].imshow(self.grid)
            axes[0].axis('off')
            axes[1].axis('off')
            axes[2].axis('off')

            plt.show()
        return sdf, sdf_gradient

    def step(self, control):
        self.state = self.A @ self.state + self.B @ control
        return self.state, self.check_collision(self.state)

    def check_collision(self, state):
        pstate = self._convert_position_to_pixels(state[:2])
        if pstate[1] < 0 or pstate[0] < 0 or pstate[1] > 63 or pstate[0] > 63:
            return True
        return self.sdf[pstate[1], pstate[0]] < 0


if __name__ == '__main__':

    env = DoubleIntegrator2DEnv()
    env.env_type = 'narrow_passages'
    env.reset()