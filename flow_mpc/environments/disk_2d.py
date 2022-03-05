"""
A pseudo 2d environment consists of a roaming disk, a maze, and several obstacles. Implemented using Mujoco
"""
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation as animation
# from IPython.display import HTML
from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors
from dm_control import viewer


class Disk(object):
    def __init__(self, r=0.1):
        self.model = mjcf.RootElement()
        self.model.compiler.angle = 'radian'
        # disk
        # rgba = np.random.uniform([0,0,0,1], [1,1,1,1])
        rgba = [0.2, 0.8, 0.2, 1]
        # self.torso = self.model.worldbody.add('body', name='torso')
        # self.torso.add('geom', name='torso_1', type='cylinder', size=[r, 0.01], pos=[0, 0, 0], rgba=rgba)
        # self.torso.add('geom', name='torso_2', type='cylinder', size=[r/1.5, 0.01], pos=[r, 0, 0], rgba=rgba)
        self.model.worldbody.add('geom', name='torso_1', type='cylinder', size=[r, 0.01], pos=[0, 0, 0.01], rgba=rgba)
        # self.model.worldbody.add('geom', name='torso_2', type='cylinder', size=[r / 1.5, 0.01], pos=[r, 0, 0.01], rgba=rgba)
        # self.torso.add('freejoint')
        # self.model.actuator.add('motor', name='x', joint=self.torso.freejoint, gear=[1,0,0,0,0,0], forcelimited=True, forcerange=[-5, 5])
        # self.model.actuator.add('motor', name='y', joint=self.torso.freejoint, gear=[0,1,0,0,0,0], forcelimited=True, forcerange=[-5, 5])
        # self.model.actuator.add('motor', name='theta', joint=self.torso.freejoint, gear=[0,0,0,0,0,1], forcelimited=True, forcerange=[-2, 2])


class Maze(object):
    def __init__(self):
        self.model = mjcf.RootElement()
        # obstacle
        # rgba = np.random.uniform([0,0,0,1], [1,1,1,1])
        rgba = [0.2, 0.2, 0.2, 1]
        self.model.worldbody.add('geom', name='left_wall', type='box', size=[0.01, 0.72, 0.1], pos=[-0.71, 0, 0.1], rgba=rgba)
        self.model.worldbody.add('geom', name='up_wall', type='box', size=[0.72, 0.01, 0.1], pos=[0.0, 0.71, 0.1], rgba=rgba)
        self.model.worldbody.add('geom', name='down_wall', type='box', size=[0.72, 0.01, 0.1], pos=[0.0, -0.71, 0.1], rgba=rgba)
        self.model.worldbody.add('geom', name='right_wall', type='box', size=[0.01, 0.72, 0.1], pos=[0.71, 0, 0.1], rgba=rgba)
        self.model.worldbody.add('geom', name='abstacle_1', type='cylinder', size=[0.05, 0.1], pos=[0.0, 0.2, 0.1], rgba=rgba)
        self.model.worldbody.add('geom', name='abstacle_2', type='cylinder', size=[0.05, 0.1], pos=[-0.3, -0.3, 0.1], rgba=rgba)
        self.model.worldbody.add('geom', name='abstacle_3', type='cylinder', size=[0.05, 0.1], pos=[0.3, -0.3, 0.1], rgba=rgba)


class DiskEntity(composer.Entity):
    """A multi-legged creature derived from `composer.Entity`."""

    def _build(self):
        self._model = Disk().model

    # def _build_observables(self):
    #     return DiskEntityObservables(self)

    @property
    def mjcf_model(self):
        return self._model

    @property
    def actuators(self):
        return tuple(self._model.find_all('actuator'))


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


class MazeEntity(composer.Entity):
    def _build(self):
        self._model = Maze().model

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
        return x, y, 0


class NavigationFixObstacle(composer.Task):
    NUM_SUBSTEPS = 25  # The number of physics substeps per control timestep.

    def __init__(self):
        self._arena = floors.Floor()
        self._robot = DiskEntity()
        self._maze = MazeEntity()
        self._arena.attach(self._maze)
        rob_frame = self._arena.add_free_entity(self._robot)
        self.rob_freejoint = rob_frame.find_all('joint')[0]

        # texture and light
        self._arena.mjcf_model.worldbody.add('light', pos=[0, 0, 3], dir=[0, 0, -1])

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
        self._mjcf_variator.apply_variations(random_state)
        # pass

    def initialize_episode(self, physics, random_state):
        valid = False
        while not valid:
            self._physics_variator.apply_variations(physics, random_state)
            robot_pose = variation.evaluate(
                (self._robot_initial_pose),
                random_state=random_state)
            robot_vel = variation.evaluate(
                (self._robot_initial_velocity),
                random_state=random_state)
            self._robot.set_pose(physics, position=robot_pose, quaternion=np.array([1, 0, 0, 0]))
            self._robot.set_velocity(physics, velocity=robot_vel, angular_velocity=np.zeros(3))
            physics.bind(self._actuators).ctrl = np.zeros(2)
            physics.step()      # execute one step collision calculation
            # collision = check_body_collision(physics, 'unnamed_model/', 'unnamed_model_1/')
            # print(f"Collision: {collision}")
            valid = not check_body_collision(physics, 'unnamed_model/', 'unnamed_model_1/')
            self._robot.set_pose(physics, position=robot_pose, quaternion=np.array([1, 0, 0, 0]))
            self._robot.set_velocity(physics, velocity=robot_vel, angular_velocity=np.zeros(3))
            physics.bind(self._actuators).ctrl = np.zeros(2)


    def get_reward(self, physics):
        # return self._button.num_activated_steps / NUM_SUBSTEPS
        return 0


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


if __name__ == "__main__":
    task = NavigationFixObstacle()
    # seed = np.random.RandomState(22)
    seed = None
    env = composer.Environment(task, random_state=seed)
    history = []
    from flow_mpc.controllers import RandomController
    controller = RandomController(udim=2, urange=10, horizon=40, lower_bound=[-10, -10], upper_bound=[10, 10])
    action_seq = controller.step(x=None)
    i = 0
    def random_policy(time_step):
        history.append(env.physics.data.time)
        # print(env.physics.data.time)
        # print(time_step)
        action_low = np.array([-20, -20])
        action_high = -action_low
        action = np.random.uniform(action_low, action_high)
        global i
        if i < len(action_seq):
            action = action_seq[i]
            i += 1
        else:
            action = np.array([0,0])
        print(action)
        return action
    # obs = env.reset()
    viewer.launch(env, policy=random_policy)


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
