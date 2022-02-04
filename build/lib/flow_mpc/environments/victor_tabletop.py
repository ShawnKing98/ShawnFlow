import sys

import numpy as np
from flow_mpc.environments.environment import Environment
from flow_mpc.environments.worlds import World
import torch
import pybullet as p
import pybullet_data
import pathlib
import pytorch_kinematics as pk
import os
import time

FLOW_MPC_ROOT = pathlib.Path(__file__).resolve().parents[1]

VICTOR_URDF_FILE = '/home/thomas/noetic_catkin_ws/src/kuka_iiwa_interface/victor_description/urdf/victor.urdf'
# VICTOR_URDF_FILE = 'home/tpower/catkin_ws/src/kuka_iiwa_interface/victor_description/urdf/victor.urdf'


class VictorTableTopWorld(World):

    def __init__(self, use_sim=False):
        self.obstacle_ids = []
        super().__init__(dim_world=3, world_size=1)
        self.table_offset = np.array([0.0, 0.0, -0.375])
        self.table_surface = -0.25
        self.table_x_range = [-0.4, 0.4]
        self.table_y_range = [-0.3, 0.3]
        self.object_ids = []
        self.use_sim = use_sim

    def _gen_table(self, occupancy_grid):
        table_w, table_l, table_h = np.random.rand(3)
        table_w = 0.9 + 0.4 * table_w
        table_l = 0.6 + 0.4 * table_l
        table_h = 0.05 + 0.15 * table_h

        table_x, table_y, table_z = np.random.rand(3)
        table_y = 0.25 * (table_y - 0.5)
        table_x = 0.25 * (table_x - 0.5)
        table_z = 0.25 * (table_z - 0.5)  # [-0.125, 0.125] -> [-0.375, -0.125

        table_centre = np.array([table_x, table_y, table_z]) + self.table_offset
        table_dims = np.array([table_l, table_w, table_h])

        self._fill_grid_w_cube(occupancy_grid, table_centre, table_dims / 2)
        if self.use_sim:
            self._add_box_to_sim(table_centre, table_dims / 2)

        # save some stuff for generating objects on table
        self.table_surface = table_z + self.table_offset[-1] + table_h / 2
        self.table_x_range = [table_centre[0] - table_l / 2, table_centre[0] + table_l / 2]
        self.table_y_range = [table_centre[1] - table_w / 2, table_centre[1] + table_w / 2]

    def _add_cubes_to_table(self, grid):
        num_boxes = np.random.randint(low=1, high=7)
        for box in range(num_boxes):
            box_w, box_l, box_h = np.random.rand(3)
            box_w = 0.3 * box_w ** 1.75 + 0.05
            box_l = 0.3 * box_l ** 1.75 + 0.05
            box_h = 0.6 * box_h ** 1.75 + 0.1

            box_x, box_y = np.random.rand(2)
            box_x = self.table_x_range[0] + (self.table_x_range[1] - self.table_x_range[0]) * box_x
            box_y = self.table_y_range[0] + (self.table_y_range[1] - self.table_y_range[0]) * box_y
            box_z = self.table_surface + box_h / 2

            box_location = np.array([box_x, box_y, box_z])
            box_halfdims = np.array([box_l, box_w, box_h]) / 2
            self._fill_grid_w_cube(grid, box_location, box_halfdims)
            if self.use_sim:
                self._add_box_to_sim(box_location, box_halfdims)

    def _add_box_to_sim(self, box_location, box_half_extents):
        rgba = np.random.rand(4)
        rgba[3] = 1
        v = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=box_half_extents,
                                rgbaColor=rgba,
                                visualFramePosition=[0, 0, 0]
                                )

        c = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=box_half_extents,
                                   collisionFramePosition=[0, 0, 0],
                                   flags=1)

        id = p.createMultiBody(baseMass=0,
                               baseInertialFramePosition=[0, 0, 0],
                               baseCollisionShapeIndex=c,
                               baseVisualShapeIndex=v,
                               basePosition=box_location,
                               useMaximalCoordinates=True)
        self.object_ids.append(id)

    def _fill_grid_w_cube(self, grid, location, half_dimensions):
        # assume world is 1x1x1m and 64x64x64 voxels
        pixel_centre = self.position_to_pixels(location)
        pixel_dims = self.distance_to_pixels(half_dimensions)

        grid[
        max(0, pixel_centre[0] - pixel_dims[1]):min(pixel_centre[0] + pixel_dims[0], 63),
        max(0, pixel_centre[1] - pixel_dims[0]):min(pixel_centre[1] + pixel_dims[1], 63),
        max(0, pixel_centre[2] - pixel_dims[2]):min(pixel_centre[2] + pixel_dims[2], 63),
        ] = 1.0

    def _get_occupancy_grid(self):
        grid = np.zeros((64, 64, 64))
        self._gen_table(grid)
        self._add_cubes_to_table(grid)
        return grid

    def reset(self):
        if self.use_sim:
            for id in self.object_ids:
                p.removeBody(id)
        super().reset()

class VictorEnv:

    def __init__(self, gui=True, debug=False, dt=0.1, world_type='spheres'):
        self.dt = dt
        self.state_dim = 7
        self.control_dim = 7
        self.gui = gui
        self.sim = None
        self.debug = debug
        self.arm_joint_idxs = [5, 6, 7, 8, 9, 10, 11]
        self.robot_base_position = np.array([-0.65, -0.25, -1])
        self.victor_joints_lower = np.array([-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239,
                                             -2.96705972839, -2.09439510239, -3.05432619099])
        self.victor_joints_upper = -self.victor_joints_lower

        self.collision_balls = []
        self._setup_camera()

        self.chain = pk.build_serial_chain_from_urdf(
            open(VICTOR_URDF_FILE).read(),
            'victor_left_gripper_palm')

        self._setup_pybullet()
        self._setup_scene()

        self.world = VictorTableTopWorld(use_sim=gui)
        self.world.reset()

    def _setup_camera(self):
        # self.viewMatrix = p.computeViewMatrix(
        #    cameraEyePosition=[1.5, 0, 1.5],
        #    cameraTargetPosition=[0, 0, 0],
        #    cameraUpVector=[-1, 0, 1])
        self.viewMatrix = p.computeViewMatrix(
            cameraEyePosition=[1.5, -0.5, 1.5],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[-np.sqrt(2) / 2.0, np.sqrt(2) / 2, 4])

        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=10.0)

    def _setup_pybullet(self):
        if self.gui:
            self.sim = p.connect(p.GUI, options="--opengl2'")
        else:
            self.sim = p.connect(p.DIRECT)

        # p.setGravity(0, 0, -9.81)
        # p.setTimeStep(self.dt)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _setup_scene(self):
        # load base & robot arm without additional stuff
        # Base for arm to sit on
        p.loadURDF("plane.urdf", basePosition=[0, 0, -1])

        # add arm to base
        self.robot = p.loadURDF(VICTOR_URDF_FILE,
                                basePosition=self.robot_base_position)
        if self.gui:
            self.goal_robot = p.loadURDF(VICTOR_URDF_FILE,
                                         basePosition=self.robot_base_position)

        # get right arm out the way
        right_arm_positions = [0.7, -2, -1, -1, 0, 0, 0]
        for idx, theta in zip(range(33, 33 + 7), right_arm_positions):
            p.resetJointState(self.robot, idx, theta)

        if self.gui:
            # get right arm out the way
            eight_arm_positions = [0.7, -2, -1, -1, 0, 0, 0]
            for idx, theta in zip(range(33, 33 + 7), right_arm_positions):
                p.resetJointState(self.goal_robot, idx, theta) \

    def step(self, control):
        # commands are joint velocities
        # TODO make the dynamics actually occur in pybullet
        self.state = self.state + self.dt * control
        self._set_state(self.state)
        # p.stepSimulation()
        # ime.sleep(self.dt)
        if self.debug:
            self.remove_collision_balls()

        collision = self.check_collision(self.state)

        if self.debug:
            self._display_collision_check_spheres()

        if self.gui:
            width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                width=224,
                height=224,
                viewMatrix=self.viewMatrix,
                projectionMatrix=self.projectionMatrix)

        return self._get_state(), collision

    def reset(self):
        self.start = None
        self.goal = None
        success = False
        while not success:
            if self.gui:
                p.resetSimulation()
                self._setup_scene()
            self.world.reset()

            success = self.reset_start_and_goal()

        self.state = self.start.copy()
        if self.gui:
            for idx, theta in zip(self.arm_joint_idxs, self.goal):
                p.resetJointState(self.goal_robot, idx, theta)

    def _set_state(self, state):
        for idx, theta in zip(self.arm_joint_idxs, state):
            p.resetJointState(self.robot, idx, theta)
        self.state = state
        self.remove_collision_balls()

    def _get_state(self):
        state = np.zeros(7)
        for i, idx in enumerate(self.arm_joint_idxs):
            state[i] = p.getJointState(self.robot, idx)[0]
        return state

    def check_collision(self, state, check_bounds=False):
        # TODO this can be done in pybullet
        look_up_points = self._get_collision_check_points(state)
        r = 0.11
        # check for self collision
        torch_points = torch.from_numpy(look_up_points)
        sq_distance = torch.cdist(torch_points, torch_points)  # + diag
        # Flags for if it's ok for these spheres to be intersecting
        allowable = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 0, 0, 0, 0, 0],
                                  [0, 1, 1, 1, 0, 0, 0, 0],
                                  [0, 0, 1, 1, 1, 0, 0, 0],
                                  [0, 0, 0, 1, 1, 1, 1, 0],
                                  [0, 0, 0, 0, 1, 1, 1, 0],
                                  [0, 0, 0, 0, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 1, 1]])

        sq_distance += allowable

        if sq_distance.min() < 2 * r:
            return True

        for idx, point in enumerate(look_up_points):
            pixels = self.world.position_to_pixels(point)
            out_of_bounds = self.world.check_bounds(pixels)

            if idx == len(look_up_points) - 1:
                if out_of_bounds and check_bounds:
                    return True

            if out_of_bounds:
                continue

            if self.world.sdf[pixels[0], pixels[1], pixels[2]] < r:
                return True

        return False

    def _get_collision_check_points(self, state):
        theta = torch.from_numpy(state)
        link_poses = self.chain.forward_kinematics(theta, end_only=False)
        offsets = torch.tensor([[0.0, 0.0, 0.0]])
        # [0.0, 0.0, -0.075]])

        look_up_points = []

        for link in link_poses.keys():
            if 'link_1' in link or 'link_6' in link or 'link_7' in link:
                points = link_poses[link].transform_points(torch.zeros(1, 3, device=offsets.device)).reshape(-1, 3)
            elif 'gripper_palm' in link:
                points = link_poses[link].transform_points(
                    torch.tensor([[0.0, 0.0, 0.1]])
                ).reshape(-1, 3)

            elif 'victor_left_arm_link' in link and '0' not in link:
                points = link_poses[link].transform_points(offsets).reshape(-1, 3)
            else:
                continue

            points = points + torch.from_numpy(self.robot_base_position)
            look_up_points.append(points)

        return torch.cat(look_up_points, dim=0).numpy()

    def _display_collision_check_spheres(self):
        r = 0.11

        collision_balls = []
        look_up_points = self._get_collision_check_points(self.state)

        for point in look_up_points:
            v = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=r,
                                    rgbaColor=[.0, 0.8, 0., 0.4],
                                    visualFramePosition=[0, 0, 0]
                                    )
            c = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=r, collisionFramePosition=[0, 0, 0], flags=1)
            b = p.createMultiBody(baseMass=0,
                                  baseInertialFramePosition=[0, 0, 0],
                                  baseVisualShapeIndex=v,
                                  baseCollisionShapeIndex=c,
                                  basePosition=point,
                                  useMaximalCoordinates=True)

            collision_balls.append(b)

        self.collision_balls = collision_balls
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=224,
            height=224,
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix)

    def remove_collision_balls(self):
        for id in self.collision_balls:
            p.removeBody(id)

        self.collision_balls = []

    def get_sdf(self):
        return self.world.sdf, self.world.sdf_grad

    def cost(self):
        return np.linalg.norm(self.state - self.goal)

    def at_goal(self):
        return self.cost() < 0.2

    def reset_start_and_goal(self):
        # Resets start and goal, returns success flag

        self.start = self.sample_config()
        if self.start is None:
            return False

        while True:
            self.goal = self.sample_config()
            if self.goal is not None:
                return True

            #min_distance = np.random.rand() * 4
            #if np.linalg.norm(self.goal - self.start) > min_distance:
            #    break

    def sample_config(self):
        # _link_name_to_index = {p.getBodyInfo(model_id)[0].decode('UTF-8'): -1, }

        # for _id in range(p.getNumJoints(model_id)):
        #    _name = p.getJointInfo(model_id, _id)[12].decode('UTF-8')
        #    _link_name_to_index[_name] = _id

        # print(_link_name_to_index)
        for _ in range(1000):

            # bias towards obstacles?
            end_effector_pos = np.random.rand(3)  # - 0.5
            # bias sampling towards lower down - where there are obstacles
            end_effector_pos[2] = end_effector_pos[2] ** 1.5
            end_effector_pos -= 0.5
            end_effector_voxels = tuple(self.world.position_to_pixels(end_effector_pos))
            distance_to_obs = self.world.sdf[end_effector_voxels[0],
                                             end_effector_voxels[1],
                                             end_effector_voxels[2]
            ]

            # accept stuff closer to obstacles with higher probability
            p_acceptance = 1.0 - distance_to_obs / np.max(self.world.sdf)
            if np.random.rand() > p_acceptance or distance_to_obs < 0.11:
                continue

            # end_effector_pos += self.robot_base_position

            # Now we will do IK
            config = p.calculateInverseKinematics(self.robot, 15, targetPosition=end_effector_pos,
                                                  lowerLimits=self.victor_joints_lower,
                                                  upperLimits=self.victor_joints_upper)
            state = np.array(config[:7])
            state[6] += np.random.randn()

            if not self.check_collision(state):
                return state

        return None


if __name__ == "__main__":
    env = VictorEnv(debug=True)
    # time.sleep(10)
    env.reset()
    # env.world.render()
    # env.reset_start_and_goal()
    for _ in range(1000):
        state, collision = env.step(np.random.randn(7))
        # if collision:
        #    print('COLLISION')
        #    time.sleep(10)
