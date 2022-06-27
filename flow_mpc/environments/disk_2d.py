"""
Added by Shawn
A pseudo 2d environment consists of a roaming disk, a maze, and several obstacles. Implemented using Mujoco
"""
import ipdb
import numpy as np
import matplotlib
import torch.nn
from matplotlib import pyplot as plt
from matplotlib import animation as animation
# from IPython.display import HTML
from dm_control import mjcf
from dm_control import composer
from dm_control import mujoco
from dm_control.composer.observation import observable
from dm_control.composer import variation
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors
from dm_control import viewer
from dm_control.composer.environment import EpisodeInitializationError

from flow_mpc.controllers import RandomController, EpsilonGreedyController


class Disk(object):
    def __init__(self, r, rgba):
        self.model = mjcf.RootElement()
        self.model.compiler.angle = 'radian'
        # disk
        # rgba = np.random.uniform([0,0,0,1], [1,1,1,1])
        # self.torso = self.model.worldbody.add('body', name='torso')
        # self.torso.add('geom', name='torso_1', type='cylinder', size=[r, 0.01], pos=[0, 0, 0], rgba=rgba)
        # self.torso.add('geom', name='torso_2', type='cylinder', size=[r/1.5, 0.01], pos=[r, 0, 0], rgba=rgba)
        self.model.worldbody.add('geom', name='torso_1', type='cylinder', size=[r, 0.01], pos=[0, 0, 0.01], rgba=rgba)
        # self.model.worldbody.add('geom', name='torso_2', type='cylinder', size=[r / 1.5, 0.01], pos=[r, 0, 0.01], rgba=rgba)
        # self.torso.add('freejoint')
        # self.model.actuator.add('motor', name='x', joint=self.torso.freejoint, gear=[1,0,0,0,0,0], forcelimited=True, forcerange=[-5, 5])
        # self.model.actuator.add('motor', name='y', joint=self.torso.freejoint, gear=[0,1,0,0,0,0], forcelimited=True, forcerange=[-5, 5])
        # self.model.actuator.add('motor', name='theta', joint=self.torso.freejoint, gear=[0,0,0,0,0,1], forcelimited=True, forcerange=[-2, 2])


class DiskEntity(composer.Entity):
    """A multi-legged creature derived from `composer.Entity`."""

    def _build(self, r=0.1, rgba=(0.2, 0.8, 0.2, 1)):
        self._model = Disk(r=r, rgba=rgba).model

    # def _build_observables(self):
    #     return DiskEntityObservables(self)

    @property
    def mjcf_model(self):
        return self._model

    # @property
    # def actuators(self):
    #     return tuple(self._model.find_all('actuator'))


# Add simple observable features for joint angles and velocities.
class DiskEntityObservables(composer.Observables):

    @composer.observable
    def joint_positions(self):
        joint = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qpos', joint)

    @composer.observable
    def joint_velocities(self):
        joint = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qvel', joint)


# class Maze(object):
#     def __init__(self):
#         self.model = mjcf.RootElement()
#         # obstacle
#         # rgba = np.random.uniform([0,0,0,1], [1,1,1,1])
#         rgba = [0.2, 0.2, 0.2, 1]
#         self.model.worldbody.add('geom', name='left_wall', type='box', size=[0.01, 0.72, 0.1], pos=[-0.71, 0, 0.1], rgba=rgba)
#         self.model.worldbody.add('geom', name='up_wall', type='box', size=[0.72, 0.01, 0.1], pos=[0.0, 0.71, 0.1], rgba=rgba)
#         self.model.worldbody.add('geom', name='down_wall', type='box', size=[0.72, 0.01, 0.1], pos=[0.0, -0.71, 0.1], rgba=rgba)
#         self.model.worldbody.add('geom', name='right_wall', type='box', size=[0.01, 0.72, 0.1], pos=[0.71, 0, 0.1], rgba=rgba)
#         self.model.worldbody.add('geom', name='obstacle_1', type='cylinder', size=[0.05, 0.1], pos=[0.04, 0.35, 0.1], rgba=rgba)
#         self.model.worldbody.add('geom', name='obstacle_2', type='cylinder', size=[0.05, 0.1], pos=[-0.37, -0.33, 0.1], rgba=rgba)
#         self.model.worldbody.add('geom', name='obstacle_3', type='cylinder', size=[0.05, 0.1], pos=[0.33, -0.22, 0.1], rgba=rgba)
#         self.model.worldbody.add('geom', name='obstacle_4', type='cylinder', size=[0.05, 0.1], pos=[0.02, -0.4, 0.1], rgba=rgba)
#         self.model.worldbody.add('geom', name='obstacle_5', type='cylinder', size=[0.05, 0.1], pos=[-0.36, 0.28, 0.1], rgba=rgba)
#         self.model.worldbody.add('geom', name='obstacle_6', type='cylinder', size=[0.05, 0.1], pos=[0.40, 0.2, 0.1], rgba=rgba)
#         self.model.worldbody.add('geom', name='obstacle_7', type='cylinder', size=[0.05, 0.1], pos=[-0.02, -0.03, 0.1], rgba=rgba)


class MazeEntity(composer.Entity):
    def _build(self, obstacle_num=10, fixed_obstacle=False):
        self._model = mjcf.RootElement()
        self._obstacle_num = obstacle_num
        self.fixed_obstacle = fixed_obstacle
        self._rgba = [0.2, 0.2, 0.2, 1]
        self._model.worldbody.add('geom', name='left_wall', type='box', size=[0.01, 0.72, 0.1], pos=[-0.71, 0, 0.1],
                                  rgba=self._rgba)
        self._model.worldbody.add('geom', name='up_wall', type='box', size=[0.72, 0.01, 0.1], pos=[0.0, 0.71, 0.1],
                                  rgba=self._rgba)
        self._model.worldbody.add('geom', name='down_wall', type='box', size=[0.72, 0.01, 0.1], pos=[0.0, -0.71, 0.1],
                                  rgba=self._rgba)
        self._model.worldbody.add('geom', name='right_wall', type='box', size=[0.01, 0.72, 0.1], pos=[0.71, 0, 0.1],
                                  rgba=self._rgba)
        self.reset_obstacles()

    def reset_obstacles(self, random_state=None):
        # first remove all the obstacles except 4 walls
        while len(self._model.worldbody.all_children()) > 4:
            self._model.worldbody.all_children()[4].remove()
        if not self.fixed_obstacle:
            # add random obstacles
            radius = distributions.Uniform(low=[0.05 for _ in range(self._obstacle_num)],
                                           high=[0.2 for _ in range(self._obstacle_num)])
            position = distributions.Uniform(low=[[-0.7, -0.7] for _ in range(self._obstacle_num)],
                                             high=[[0.7, 0.7] for _ in range(self._obstacle_num)])
            r = radius(random_state=random_state)
            p = position(random_state=random_state)
            for i in range(self._obstacle_num):
                self._model.worldbody.add('geom', name=f'obstacle_{i}', type='cylinder', size=[r[i], 0.1],
                                          pos=[*p[i], 0.1], rgba=self._rgba)
        else:
            self._model.worldbody.add('geom', name='obstacle_1', type='cylinder', size=[0.05, 0.1],
                                      pos=[0.04, 0.35, 0.1], rgba=self._rgba)
            self._model.worldbody.add('geom', name='obstacle_2', type='cylinder', size=[0.05, 0.1],
                                      pos=[-0.37, -0.33, 0.1], rgba=self._rgba)
            self._model.worldbody.add('geom', name='obstacle_3', type='cylinder', size=[0.05, 0.1],
                                      pos=[0.33, -0.22, 0.1], rgba=self._rgba)
            self._model.worldbody.add('geom', name='obstacle_4', type='cylinder', size=[0.05, 0.1],
                                      pos=[0.02, -0.4, 0.1], rgba=self._rgba)
            self._model.worldbody.add('geom', name='obstacle_5', type='cylinder', size=[0.05, 0.1],
                                      pos=[-0.36, 0.28, 0.1], rgba=self._rgba)
            self._model.worldbody.add('geom', name='obstacle_6', type='cylinder', size=[0.05, 0.1],
                                      pos=[0.40, 0.2, 0.1], rgba=self._rgba)
            self._model.worldbody.add('geom', name='obstacle_7', type='cylinder', size=[0.05, 0.1],
                                      pos=[-0.02, -0.03, 0.1], rgba=self._rgba)

    @property
    def mjcf_model(self):
        return self._model


class UniformBox(variation.Variation):
    """A uniformly sampled horizontal point on a circle of radius `distance`."""

    def __init__(self, x_range, y_range):
        self._x_range = x_range
        self._y_range = y_range

    def __call__(self, initial_value=None, current_value=None, random_state=None):
        x, y = variation.evaluate(
            (self._x_range, self._y_range), random_state=random_state)
        return x, y, 0.


class NavigationObstacle(composer.Task):
    NUM_SUBSTEPS = 50  # The number of physics substeps per control timestep. # Default physics substep takes 2ms

    def __init__(self, action_noise: float = 0.0, process_noise: float = 0.0, fixed_obstacle=False, cost_function=None):
        self._arena = floors.Floor()
        self._camera = self._arena.mjcf_model.find_all('camera')[0]
        self._robot = DiskEntity()
        self._goal = DiskEntity(r=0.05, rgba=(0.8, 0.2, 0.2, 1))
        self._maze = MazeEntity(fixed_obstacle=fixed_obstacle)
        self._arena.attach(self._maze)
        rob_frame = self._arena.add_free_entity(self._robot)
        goal_frame = self._arena.add_free_entity(self._goal)
        self.rob_freejoint = rob_frame.find_all('joint')[0]
        self.goal_freejoint = goal_frame.find_all('joint')[0]
        self.action_noise = action_noise
        self.process_noise = process_noise
        # self._goal_indicator = self._arena.mjcf_model.worldbody.add('geom', name='goal', type='cylinder', size=[0.05, 0.01], pos=[0., 0., -0.01], rgba=[0.9, 0.1, 0.1, 1])
        # self._goal_cache = None
        if cost_function is None:
            self.cost_function = NaiveCostFunction([0., 0.])
        else:
            self.cost_function = cost_function

        # texture and light
        self._arena.mjcf_model.worldbody.add('light', pos=[0, 0, 3], dir=[0, 0, -1])

        # camera pose
        self._camera.pos = [0, 0, 10]

        # actuators
        self._arena.mjcf_model.actuator.add('motor', name='x', joint=self.rob_freejoint, gear=[1, 0, 0, 0, 0, 0],
                                            forcelimited=True, forcerange=[-10, 10])
        self._arena.mjcf_model.actuator.add('motor', name='y', joint=self.rob_freejoint, gear=[0, 1, 0, 0, 0, 0],
                                            forcelimited=True, forcerange=[-10, 10])
        # self._arena.mjcf_model.actuator.add('motor', name='theta', joint=self.rob_freejoint, gear=[0, 0, 0, 0, 0, 1],
        #                                     forcelimited=True, forcerange=[-2, 2])
        self._actuators = self._arena.mjcf_model.find_all('actuator')

        # Configure initial poses
        self._x_range = distributions.Uniform(-0.7, 0.7)
        self._y_range = distributions.Uniform(-0.7, 0.7)
        self._robot_initial_pose = UniformBox(self._x_range, self._y_range)
        self._vx_range = distributions.Uniform(-3, 3)
        self._vy_range = distributions.Uniform(-3, 3)
        self._robot_initial_velocity = UniformBox(self._x_range, self._y_range)
        self._goal_x_range = distributions.Uniform(-0.7, 0.7)
        self._goal_y_range = distributions.Uniform(-0.7, 0.7)
        self._goal_generator = UniformBox(self._goal_x_range, self._goal_y_range)

        # Configure variators (for randomness)
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        def robot_position(physics):
            return physics.bind(self.rob_freejoint).qpos

        def robot_velocity(physics):
            return physics.bind(self.rob_freejoint).qvel

        self._task_observables = {}
        self._task_observables['robot_position'] = observable.Generic(robot_position)
        self._task_observables['robot_velocity'] = observable.Generic(robot_velocity)

        # Configure and enable observables
        pos_corruptor = noises.Additive(distributions.Normal(scale=0.01))
        # pos_corruptor = None
        # self._task_observables['robot_position'].corruptor = pos_corruptor
        self._task_observables['robot_position'].enabled = True
        vel_corruptor = noises.Multiplicative(distributions.LogNormal(sigma=0.01))
        # vel_corruptor = None
        # self._task_observables['robot_velocity'].corruptor = vel_corruptor
        self._task_observables['robot_velocity'].enabled = True
        # self._button.observables.touch_force.enabled = True

        for obs in self._task_observables.values():
            obs.enabled = True

        self.control_timestep = self.NUM_SUBSTEPS * self.physics_timestep

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode_mjcf(self, random_state):
        # self._mjcf_variator.apply_variations(random_state)
        self._maze.reset_obstacles(random_state)
        # if self._goal_cache is None:
        #     self._maze.reset_obstacles(random_state)
        # else:
        #     self._goal_indicator.pos[0:2] = self._goal_cache
        #     self.cost_function.goal = self._goal_cache

    def initialize_episode(self, physics, random_state):
        while True:
            # self._physics_variator.apply_variations(physics, random_state)
            robot_pose, robot_vel, goal = variation.evaluate(
                (self._robot_initial_pose, self._robot_initial_velocity, self._goal_generator),
                random_state=random_state)
            with physics.reset_context():
                self._robot.set_pose(physics, position=robot_pose, quaternion=np.array([1, 0, 0, 0]))
                self._robot.set_velocity(physics, velocity=robot_vel, angular_velocity=np.zeros(3))
                self._goal.set_pose(physics, position=goal, quaternion=np.array([1, 0, 0, 0]))
                # self._robot.set_pose(physics, position=goal, quaternion=np.array([1, 0, 0, 0]))
            if check_body_collision(physics, 'unnamed_model/', 'unnamed_model_1/') \
                    or check_body_collision(physics, 'unnamed_model/', 'unnamed_model_2/') \
                    or check_body_collision(physics, 'unnamed_model_1/', 'unnamed_model_2/'):
                continue
            break
        self.cost_function.goal = goal[0:2]
        aerial_image = self.get_aerial_view(physics)
        self.cost_function.env_image = aerial_image

        # print(f"Robot pose: {physics.bind(self.rob_freejoint).qpos[0:3]}")

        # if self._goal_cache is None:
        #     self._goal_cache = goal[0:2]
        #     raise EpisodeInitializationError
        # else:
        #     self._goal_cache = None

        # init_pos = physics.bind(self.rob_freejoint).qpos[0:3]
        # print(f"initial position: {init_pos}")
        # self._goal_indicator.pos = [goal[0], goal[1], -0.01]
        # self._goal_indicator = self._arena.mjcf_model.worldbody.add('geom', name='goal', type='cylinder', size=[0.05, 0.01], pos=[goal[0], goal[1], -0.01], rgba=[0.9, 0.1, 0.1, 1])
        # print("final robot pose:", self.observables['robot_position'](physics)[0:2], "desired:", goal[0:2])
        # print("final goal pose:", self._goal_indicator.pos[0:2], "desired:", goal[0:2])
        # print("-"*10)
        # print("Final goal:", self.goal)
        # print("Obstacle position:", self._maze.mjcf_model.worldbody.all_children()[4].pos

    def before_step(self, physics, action, random_state):
        action_noise = distributions.Normal(scale=self.action_noise)
        action = action + action_noise(random_state=random_state)
        physics.set_control(action)

    def after_step(self, physics, random_state):
        # print(check_body_collision(physics, 'unnamed_model/', 'unnamed_model_1/'))
        robot_pos = physics.bind(self.rob_freejoint).qpos
        robot_vel = physics.bind(self.rob_freejoint).qvel
        # original_robot_pos = robot_pos.copy()
        # original_robot_vel = robot_vel.copy()
        pos_noise = distributions.Normal(scale=self.process_noise)
        vel_noise = distributions.LogNormal(sigma=self.process_noise)
        robot_pos[0:2] = robot_pos[0:2] + pos_noise(random_state=random_state)
        robot_vel[0:2] = robot_vel[0:2] * vel_noise(random_state=random_state)

    def get_reward(self, physics):
        # return self._button.num_activated_steps / NUM_SUBSTEPS
        collision = check_body_collision(physics, 'unnamed_model/', 'unnamed_model_1/')
        return (collision,)

    def get_aerial_view(self, physics, show=False) -> np.ndarray:
        # move the robot and the goal out of camera view temporarily
        origin_rob_pos = physics.bind(self.rob_freejoint).qpos[0:3].copy()
        origin_rob_vel = physics.bind(self.rob_freejoint).qvel[0:3].copy()
        origin_goal_pos = physics.bind(self.goal_freejoint).qpos[0:3].copy()
        with physics.reset_context():
            self._robot.set_pose(physics, position=[999, 999, 10], quaternion=np.array([1, 0, 0, 0]))
            self._goal.set_pose(physics, position=[999, 999, 10], quaternion=np.array([1, 0, 0, 0]))
        camera = mujoco.Camera(physics, height=128, width=128, camera_id='top_camera')
        seg = camera.render(segmentation=True)
        # Display the contents of the first channel, which contains object
        # IDs. The second channel, seg[:, :, 1], contains object types.
        geom_ids = seg[:, :, 0]
        # clip to bool variables
        pixels = geom_ids.clip(min=0, max=1)  # shape (height, width)
        # draw
        if show:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(pixels, cmap='gray')
        # move the robot and the goal back
        with physics.reset_context():
            self._robot.set_pose(physics, position=origin_rob_pos, quaternion=np.array([1, 0, 0, 0]))
            self._robot.set_velocity(physics, velocity=origin_rob_vel, angular_velocity=np.zeros(3))
            self._goal.set_pose(physics, position=origin_goal_pos, quaternion=np.array([1, 0, 0, 0]))
        return pixels


def check_body_collision(physics: mjcf.Physics, body1: str, body2: str):
    """
    Check whether the given two bodies have collided
    :param physics: a MuJoCo physics engine
    :param body1: the name of body1
    :param body2: the name of body2
    :return collision: a bool variable
    """
    collision = False
    bodyid1 = physics.model.name2id(body1, 'body')
    bodyid2 = physics.model.name2id(body2, 'body')
    for geom1, geom2 in zip(physics.data.contact.geom1, physics.data.contact.geom2):
        if physics.model.geom_bodyid[geom1] == bodyid1 and physics.model.geom_bodyid[geom2] == bodyid2:
            collision = True
            break
        if physics.model.geom_bodyid[geom1] == bodyid2 and physics.model.geom_bodyid[geom2] == bodyid1:
            collision = True
            break
    return collision


# def display_video(frames, framerate=30):
#     height, width, _ = frames[0].shape
#     dpi = 70
#     orig_backend = matplotlib.get_backend()
#     matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
#     fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
#     matplotlib.use(orig_backend)  # Switch back to the original backend.
#     ax.set_axis_off()
#     ax.set_aspect('equal')
#     ax.set_position([0, 0, 1, 1])
#     im = ax.imshow(frames[0])
#     def update(frame):
#       im.set_data(frame)
#       return [im]
#     interval = 1000/framerate
#     anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
#                                    interval=interval, blit=True, repeat=False)
#     plt.show()
#     # return HTML(anim.to_html5_video())
#     return

class NaiveCostFunction:
    """
    A simple cost function that encourages the robot to take the action towards the goal
    """
    def __init__(self, goal: list):
        self.goal = np.array(goal)

    def __call__(self, x: list, u: list):
        dist = self.goal - np.array(x)
        dist_norm = dist / np.linalg.norm(dist)
        u_norm = u / np.linalg.norm(u)
        return -dist_norm @ u_norm


if __name__ == "__main__":
    task = NavigationObstacle(process_noise=0.000, action_noise=0.05, fixed_obstacle=False)
    # seed = np.random.RandomState(22)
    seed = None
    env = composer.Environment(task, random_state=seed, max_reset_attempts=2)
    # obs = env.reset()

    # def mppi_plan(time_step):
    #     robot_pos = time_step.observation['robot_position'][0, 0:2]
    #     robot_vel = time_step.observation['robot_velocity'][0, 0:2]
    #     robot_state = np.concatenate((robot_pos, robot_vel), axis=0)
    #     action_seq = planner.step(robot_state)
    #     final_action = action_seq[0].cpu().detach().numpy()
    #     print(final_action)
    #     return final_action

    # viewer.launch(env, policy=mppi_plan)

    # history = []
    controller = RandomController(udim=2, urange=10, horizon=40, lower_bound=[-10, -10], upper_bound=[10, 10])
    # epsilon_controller = EpsilonGreedyController(cost=task.cost_function, lower_bound=[-10, -10], upper_bound=[10, 10])
    action_seq = controller.step(x=None)
    i = 0
    def random_policy(time_step):
        global i, action_seq
        # ipdb.set_trace()
        # history.append(env.physics.data.time)
        if env.physics.data.time < 0.02:
            i = 0
            # action_seq = controller.step(x=None)
            # robot_pos = env.physics.bind(env._task.rob_freejoint).qpos
            # robot_vel = env.physics.bind(env._task.rob_freejoint).qvel
            # robot_pos[0:2] = [0.1, 0.1]
            # robot_vel[0:2] = [0, 0]
        # print(time_step.reward)

        if i < len(action_seq):
            action = action_seq[i]
            i += 1
        else:
            action = np.array([0, 0])
            # action_seq = controller.step(x=None)
            # i = 0
        # print(f"{env.physics.data.time}s: {action}")

        # print("Real goal pos:", env._task._goal_indicator.pos[0:2], "Desired goal pose:", env._task.goal[0:2])
        # print(action)
        return action


    # task.get_aerial_view(env.physics, show=True)

    viewer.launch(env, policy=random_policy)
    # viewer.launch(env, policy=epsilon_controller.step)
    ipdb.set_trace()

    # duration = 10  # (Seconds)
    # framerate = 30  # (Hz)
    # video = [env.physics.render().copy()]
    # while env.physics.data.time < duration:
    #     action = np.random.uniform(action_low, action_high)
    #     env.step(action)
    #     if len(video) < env.physics.data.time * framerate:
    #         pixels = env.physics.render()
    #         video.append(pixels.copy())
    #
    # display_video(video, framerate)
