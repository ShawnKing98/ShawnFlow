from flow_mpc.environments.worlds import *


class Environment:
    """
    Base class for environments
    """

    def __init__(self, state_dim, control_dim, world_dim, world_type):
        self.state_dim = state_dim
        self.control_dim = control_dim

        if world_type == 'spheres':
            if world_dim == 3:
                min_obstacles = 4
                max_obstacles = 12
                min_radius = 0.4
                max_radius = 0.8
            else:
                # min_obstacles = 3
                # max_obstacles = 9
                min_obstacles = 0
                max_obstacles = 1
                min_radius = 0.35
                max_radius = 0.6
            self.world = SphereWorld(world_dim=world_dim, min_obstacles=min_obstacles, max_obstacles=max_obstacles,
                                     min_radius=min_radius, max_radius=max_radius)
        elif world_type == 'squares':
            self.world = SquareWorld(world_dim=world_dim)
        elif world_type == 'narrow_passages':
            self.world = NarrowPassagesWorld(world_dim=world_dim)
        else:
            raise ValueError('Invalid world type')

        self.state = np.zeros(self.state_dim)
        self.start = np.zeros(self.state_dim)
        self.goal = np.zeros(self.state_dim)
        self.world_type = world_type

    def reset(self):
        self.world.reset()
        self.reset_start_and_goal()

    def reset_start_and_goal(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def step(self, control):
        raise NotImplementedError

    def get_sdf(self):
        return self.world.sdf, self.world.sdf_grad

