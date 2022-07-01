import time

from dm_control import mjcf
from dm_control import composer
from dm_control import mujoco
from dm_control import viewer
from dm_control.composer import variation
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import floors
from flow_mpc.controllers import RandomController
from flow_mpc.utils import check_body_collision
from dm_env import StepType

import numpy as np
import matplotlib.pyplot as plt
import ipdb

seed = 0
# np.random.seed(seed)


class RopeEntity(composer.Entity):
    def _build(self, length=20, rgba=(0.2, 0.8, 0.2, 1), friction_noise=0.):
        self._model = mjcf.RootElement('rope')
        self._model.compiler.angle = 'radian'
        # self._model.default.geom.friction = [1, 0.1, 0.1]
        body = self._model.worldbody.add('body', name='rB0')
        self._composite = body.add('composite', prefix="r", type='rope', count=[length, 1, 1], spacing=0.084)
        self._composite.add('joint', kind='main', damping=0.01, stiffness=0.01)
        self._composite.geom.set_attributes(type='capsule', size=[0.0375, 0.04], rgba=rgba, mass=0.005,
                                            contype=1, conaffinity=1, priority=1, friction=[0.1, 5e-3, 1e-4])
        self._friction_noise_generator = distributions.LogNormal(sigma=[friction_noise for _ in range(3)])
        self.reset_friction()

    @property
    def mjcf_model(self):
        return self._model

    def reset_friction(self, random_state=None):
        friction_noise = self._friction_noise_generator(random_state=random_state)
        new_friction = np.array([1, 5e-3, 1e-4]) * friction_noise
        self._composite.geom.set_attributes(friction=new_friction)

    def reset_pose(self):
        pass


class GripperEntity(composer.Entity):
    def _build(self, name=None, rgba=(0, 1, 1, 1), xrange=(-10, 10), yrange=(-10, 10), zrange=(-0., 0.000000001), mass=0.01):
        self._model = mjcf.RootElement(name)
        body = self._model.worldbody.add('body', name='dummy')
        body.add('geom', size=[0.04, 0.04, 0.03], mass=mass, rgba=rgba, type='box', contype=2, conaffinity=2, group=1)
        body.add('joint', axis=[1, 0, 0], type='slide', name='x', pos=[0, 0, 0], limited=True, range=xrange)
        body.add('joint', axis=[0, 1, 0], type='slide', name='y', pos=[0, 0, 0], limited=True, range=yrange)
        body.add('joint', axis=[0, 0, 1], type='slide', name='z', pos=[0, 0, 0], limited=True, range=zrange)

    @property
    def mjcf_model(self):
        return self._model


class TestBox(composer.Entity):
    def _build(self, name=None, rgba=(0, 1, 1, 1), mass=0.01):
        self._model = mjcf.RootElement(name)
        # body = self._model.worldbody.add('body', name='dummy')
        # body.add('geom', size=[0.04, 0.04, 0.04], mass=mass, rgba=rgba, type='box', contype=1, conaffinity=1, group=1)
        self._model.worldbody.add('geom', size=[0.03, 0.03, 0.03], mass=mass, rgba=rgba, type='box', contype=1, conaffinity=1, group=1)

    @property
    def mjcf_model(self):
        return self._model


class MazeEntity(composer.Entity):
    def _build(self, obstacle_num=25, fixed_obstacle=False):
        self._model = mjcf.RootElement("maze")
        self._model.default.geom.contype = 3
        self._model.default.geom.conaffinity = 3
        self._obstacle_num = obstacle_num
        self.fixed_obstacle = fixed_obstacle
        self._rgba = [0.2, 0.2, 0.2, 1]
        self._model.worldbody.add('geom', name='left_wall', type='box', size=[0.01, 1.43, 0.1], pos=[-1.42, 0, 0.1],
                                  rgba=self._rgba)
        self._model.worldbody.add('geom', name='up_wall', type='box', size=[1.43, 0.01, 0.1], pos=[0.0, 1.42, 0.1],
                                  rgba=self._rgba)
        self._model.worldbody.add('geom', name='down_wall', type='box', size=[1.43, 0.01, 0.1], pos=[0.0, -1.42, 0.1],
                                  rgba=self._rgba)
        self._model.worldbody.add('geom', name='right_wall', type='box', size=[0.01, 1.43, 0.1], pos=[1.42, 0, 0.1],
                                  rgba=self._rgba)
        self.reset_obstacles()

    def reset_obstacles(self, random_state=None):
        # first remove all the obstacles except 4 walls
        while len(self._model.worldbody.all_children()) > 4:
            self._model.worldbody.all_children()[4].remove()
        if not self.fixed_obstacle:
            # add random obstacles
            radius = distributions.Uniform(low=[0.1 for _ in range(self._obstacle_num)],
                                           high=[0.2 for _ in range(self._obstacle_num)])
            position = distributions.Uniform(low=[[-1.4, -1.4] for _ in range(self._obstacle_num)],
                                             high=[[1.4, 1.4] for _ in range(self._obstacle_num)])
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


class RopeManipulation(composer.Task):
    NUM_SUBSTEPS = 100   # The number of physics substeps per control timestep. Default physics substep takes 1ms.

    def __init__(self, rope_length=11, action_noise: float = 0.0, friction_noise: float = 0.2):
        # root entity
        self._arena = floors.Floor()

        # simulation setting
        self._arena.mjcf_model.compiler.inertiafromgeom = True
        self._arena.mjcf_model.default.joint.damping = 0
        self._arena.mjcf_model.default.joint.stiffness = 0
        self._arena.mjcf_model.default.geom.contype = 3
        self._arena.mjcf_model.default.geom.conaffinity = 3
        self._arena.mjcf_model.default.geom.friction = [1, 0.1, 0.1]
        self._arena.mjcf_model.option.gravity = [1e-5, 0, -9.81]
        self._arena.mjcf_model.option.integrator = 'Euler'    # RK4 or Euler
        self._arena.mjcf_model.option.timestep = 0.001
        # self._arena.mjcf_model.option.viscosity = 1
        # self._arena.mjcf_model.size.nstack = 30000

        # other entities
        self._rope = RopeEntity(length=rope_length, friction_noise=friction_noise)
        # self._goal = DiskEntity(r=0.05, rgba=(0.8, 0.2, 0.2, 1))
        self._gripper1 = GripperEntity(name='gripper1', rgba=(0, 1, 1, 1), mass=0.01)
        # self._gripper2 = GripperEntity(name='gripper2', rgba=(0.5, 0, 0.5, 1), mass=0.1)
        self._maze = MazeEntity(fixed_obstacle=False)
        self._maze_duplicate = MazeEntity(fixed_obstacle=False)     # for getting environment image

        rope_site = self._arena.add_free_entity(self._rope)
        gripper1_site = self._arena.attach(self._gripper1)
        self._arena.attach(self._maze)
        maze_duplicate_site = self._arena.attach(self._maze_duplicate)
        # gripper2_site = self._arena.attach(self._gripper2)
        # gripper1_site = self._arena.add_free_entity(self._gripper1)
        # gripper2_site = self._arena.add_free_entity(self._gripper2)
        rope_site.pos = [0.075, 0, 0.0375]
        gripper1_site.pos = [0.0, 0, 0.0375]
        maze_duplicate_site.pos = [5, 0, 0]
        # gripper2_site.pos = [0.5, 0, 1]
        # goal_frame = self._arena.add_free_entity(self._goal)
        # self.rob_freejoint = rob_frame.find_all('joint')[0]
        # self.goal_freejoint = goal_frame.find_all('joint')[0]

        # constraint
        self._arena.mjcf_model.equality.add('connect', body1='gripper1/dummy', body2='rope/rB0', anchor=[0, 0, 0.0])
        # self._arena.mjcf_model.equality.add('connect', body1='gripper2/dummy', body2='rope/rB10', anchor=[0, 0, 0.0])

        # noise
        self.action_noise = action_noise
        self.friction_noise = friction_noise
        # if cost_function is None:
        #     self.cost_function = NaiveCostFunction([0., 0.])
        # else:
        #     self.cost_function = cost_function

        # texture and light
        self._arena.mjcf_model.worldbody.add('light', pos=[0, 0, 3], dir=[0, 0, -1])

        # camera
        top_camera = self._arena.mjcf_model.find_all('camera')[0]
        top_camera.pos = [0, 0, 20]
        env_camera = self._arena.mjcf_model.worldbody.add('camera', name='env_camera', fovy=top_camera.fovy, pos=[5, 0, 20])

        # actuators
        self._arena.mjcf_model.actuator.add('position', name='left_x', joint=self._gripper1.mjcf_model.find_all('joint')[0],
                                            ctrllimited=True, ctrlrange=[-2, 2], kp=0.5)
        self._arena.mjcf_model.actuator.add('position', name='left_y', joint=self._gripper1.mjcf_model.find_all('joint')[1],
                                            ctrllimited=True, ctrlrange=[-2, 2], kp=0.5)
        self._arena.mjcf_model.actuator.add('position', name='left_z', joint=self._gripper1.mjcf_model.find_all('joint')[2],
                                            ctrllimited=True, ctrlrange=[-2, 2], kp=0.5)
        # self._arena.mjcf_model.actuator.add('motor', name='left_x', joint=self._gripper1.mjcf_model.find_all('joint')[0],
        #                                     forcelimited=True, forcerange=[-20, 20])
        # self._arena.mjcf_model.actuator.add('motor', name='left_y', joint=self._gripper1.mjcf_model.find_all('joint')[1],
        #                                     forcelimited=True, forcerange=[-20, 20])
        # self._arena.mjcf_model.actuator.add('motor', name='left_z', joint=self._gripper1.mjcf_model.find_all('joint')[2],
        #                                     forcelimited=True, forcerange=[-20, 20])
        self._actuators = self._arena.mjcf_model.find_all('actuator')

        # Configure initial poses
        self._xy_range = distributions.Uniform(-1., 1.)
        self._joint_range = distributions.Uniform(-1, 1)
        # self._x_range = distributions.Uniform(-0.7, 0.7)
        # self._y_range = distributions.Uniform(-0.7, 0.7)
        # self._rope_initial_pose = UniformBox(self._x_range, self._y_range)
        # self._vx_range = distributions.Uniform(-3, 3)
        # self._vy_range = distributions.Uniform(-3, 3)
        # self._rope_initial_velocity = UniformBox(self._x_range, self._y_range)
        # self._goal_x_range = distributions.Uniform(-0.7, 0.7)
        # self._goal_y_range = distributions.Uniform(-0.7, 0.7)
        # self._goal_generator = UniformBox(self._goal_x_range, self._goal_y_range)

        # Configure variators (for randomness)
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        self._task_observables = {}
        self._task_observables['rope_pos'] = observable.MujocoFeature('geom_xpos', [f'rope/rG{i}' for i in range(rope_length)])

        # Configure and enable observables
        # pos_corruptor = noises.Additive(distributions.Normal(scale=0.01))
        # pos_corruptor = None
        # self._task_observables['robot_position'].corruptor = pos_corruptor
        # self._task_observables['robot_position'].enabled = True
        # vel_corruptor = noises.Multiplicative(distributions.LogNormal(sigma=0.01))
        # vel_corruptor = None
        # self._task_observables['robot_velocity'].corruptor = vel_corruptor
        # self._task_observables['robot_velocity'].enabled = True

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
        # random_state = np.random.RandomState(0)
        self._rope.reset_friction()
        # self._rope.reset_friction(random_state)
        self._maze.reset_obstacles(random_state)
        source_geoms = self._maze.mjcf_model.worldbody.find_all('geom')
        target_geoms = self._maze_duplicate.mjcf_model.worldbody.find_all('geom')
        for i in range(len(source_geoms)):
            target_geoms[i].set_attributes(**source_geoms[i].get_attributes())

    def initialize_episode(self, physics, random_state):
        # pass
        # random_state = np.random.RandomState(0)
        while True:
            x, y = self._xy_range(initial_value=np.zeros(2), random_state=random_state)
            joints = self._joint_range(initial_value=np.zeros(10), random_state=random_state)
            with physics.reset_context():
                self._rope.set_pose(physics, position=(x, y, 0.0375))
                # physics.bind(self._gripper1.mjcf_model.find_all('joint')).qpos = np.array([x-0.075, y, 0])
                self._gripper1.set_pose(physics, position=(x-0.075, y, 0.0375))
                for i in range(10):
                    physics.named.data.qpos[f'rope/rJ1_{i+1}'] = joints[i]
            if check_body_collision(physics, 'maze', 'rope') or check_body_collision(physics, 'maze', 'gripper'):
                continue
            else:
                break


    def before_step(self, physics, action, random_state):
        action_noise = distributions.Normal(scale=self.action_noise)
        action = action + action_noise(random_state=random_state)
        physics.set_control(action)

    def after_step(self, physics, random_state):
        pass
        # print(check_body_collision(physics, 'unnamed_model/', 'unnamed_model_1/'))
        # robot_pos = physics.bind(self.rob_freejoint).qpos
        # robot_vel = physics.bind(self.rob_freejoint).qvel
        # original_rope_pos = robot_pos.copy()
        # original_rope_vel = robot_vel.copy()
        # pos_noise = distributions.Normal(scale=self.process_noise)
        # vel_noise = distributions.LogNormal(sigma=self.process_noise)
        # robot_pos[0:2] = robot_pos[0:2] + pos_noise(random_state=random_state)
        # robot_vel[0:2] = robot_vel[0:2] * vel_noise(random_state=random_state)

    def get_reward(self, physics):
        # return self._button.num_activated_steps / NUM_SUBSTEPS
        # collision = check_body_collision(physics, 'unnamed_model/', 'unnamed_model_1/')
        # return (collision,)
        return 0

    def get_aerial_view(self, physics, show=False) -> np.ndarray:
        # pass
        # # move the robot and the goal out of camera view temporarily
        # origin_qpos = physics.data.qpos.copy()
        # origin_qvel = physics.data.qvel.copy()
        # # origin_rope_pos = physics.bind(mjcf.get_frame_freejoint(self._rope.mjcf_model)).qpos.copy()
        # # origin_rope_vel = physics.bind(mjcf.get_frame_freejoint(self._rope.mjcf_model)).qvel.copy()
        # origin_gripper_pos = physics.bind(mjcf.get_attachment_frame(self._gripper1.mjcf_model)).pos.copy()
        # # origin_gripper_pos = physics.bind(self._gripper1.mjcf_model.find_all('joint')).qpos.copy()
        # # origin_gripper_vel = physics.bind(self._gripper1.mjcf_model.find_all('joint')).qvel.copy()
        # with physics.reset_context():
        #     self._rope.set_pose(physics, position=[999, 999, 10])
        #     self._gripper1.set_pose(physics, position=[999-0.075, 999, 10])
        #     # physics.bind(self._gripper1.mjcf_model.find_all('joint')).qpos = np.array([999, 999, 10])
        camera = mujoco.Camera(physics, height=128, width=128, camera_id='env_camera')
        seg = camera.render(segmentation=True)
        # Display the contents of the first channel, which contains object
        # IDs. The second channel, seg[:, :, 1], contains object types.
        geom_ids = seg[:, :, 0]
        # clip to bool variables
        pixels = geom_ids.clip(min=0, max=1)  # shape (height, width)
        # draw
        if show:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(1-pixels, cmap='gray')
        # # move the robot and the goal back
        # with physics.reset_context():
        #     self._gripper1.set_pose(physics, position=origin_gripper_pos)
        #     physics.data.qpos[0:7] = origin_qpos[0:7]
        #     physics.data.qpos[7:-3] = origin_qpos[7:-3]
        #     physics.data.qpos[-3:] = origin_qpos[-3:]
        #     physics.data.qvel[0:7] = origin_qvel[0:7]
        #     physics.data.qvel[7:-3] = origin_qvel[7:-3]
        #     physics.data.qvel[-3:] = origin_qvel[-3:]
        #     # self._rope.set_pose(physics, position=origin_rope_pos[0:3], quaternion=origin_rope_pos[3:])
        #     # self._rope.set_velocity(physics, velocity=origin_rope_vel[0:3], angular_velocity=origin_rope_vel[3:])
        #
        #     # self._gripper1.set_velocity(physics, velocity=origin_gripper_vel)
        #     # physics.bind(self._gripper1.mjcf_model.find_all('joint')).qpos = origin_gripper_pos
        #     # physics.bind(self._gripper1.mjcf_model.find_all('joint')).qvel = origin_gripper_vel
        return pixels

if __name__ == "__main__":
    # r = RopeEntity()
    # static_model = """
    # <mujoco>
    #   <worldbody>
    #     <light name="top" pos="0 0 1"/>
    #     <geom conaffinity="3" condim="3" name="floor" pos="0 0 0" rgba="0 0.5 0 1" size="5 5 1" type="plane"/>
    #     <camera name="rgb" mode="fixed" pos="0 0 4" euler="0 0 0"/>
    #     <camera name="viewer1" mode="fixed" pos="3 3 4" quat="0.8535534 -0.3535534 0.1464466 -0.3535534 "/>
    #     <camera name="viewer2" mode="fixed" pos="-3 3 4" quat="0.8535534 -0.3535534 -0.1464466 0.3535534 "/>
    #     <body name="B0" pos="-.45 0 0.1">
    #         <freejoint name="r1" />
    #         <composite type="rope" count="11 1 1" spacing=".084">
    #             <joint kind="main" damping="0." stiffness=".0"/>
    #             <geom type="capsule" size=".0375 .04" rgba=" 1 0 0 1" mass="0.01" contype="2" conaffinity="2"/>
    #         </composite>
    #     </body>
    #   </worldbody>
    # </mujoco>
    # """
    # physics = mujoco.Physics.from_xml_string(static_model)
    # arena = floors.Floor()
    r = RopeEntity()
    task = RopeManipulation()
    seed = None
    env = composer.Environment(task, random_state=seed)
    obs = env.reset()

    def dummy_controller(timestep):
        print(timestep.observation['pos_0'])
        print("------------------")
        return 0


    controller = RandomController(udim=3, urange=1, horizon=40, sigma=20, lower_bound=[-1, -1, -1], upper_bound=[1, 1, 1])
    action_seq = controller.step(None)
    i = 0
    image = None
    def random_policy(time_step):
        # time.sleep(0.1)
        global i, action_seq, image
        if time_step.step_type == StepType.FIRST:
            # env._random_state.seed(0)
            # print(time_step.observation)
            action_seq = controller.step(x=None)
            i = 0
            # robot_pos = env.physics.bind(env._task.rob_freejoint).qpos
            # robot_vel = env.physics.bind(env._task.rob_freejoint).qvel
            # robot_pos[0:2] = [0.1, 0.1]
            # robot_vel[0:2] = [0, 0]
        # print(time_step.reward)

        if i < len(action_seq):
            action = action_seq[i]
            i += 1
        else:
            action = 0
            # action_seq = controller.step(x=None)
            # i = 0
        # print(f"{env.physics.data.time}s: {action}")

        # print("Real goal pos:", env._task._goal_indicator.pos[0:2], "Desired goal pose:", env._task.goal[0:2])
        # print(action)
        if i == 20:
            image = env.task.get_aerial_view(env.physics, show=False)
        print(action)
        return action

    viewer.launch(env, policy=random_policy)
    # ipdb.set_trace()
