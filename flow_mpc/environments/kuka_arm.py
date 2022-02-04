import numpy as np
from flow_mpc.environments.environment import Environment
from flow_mpc.environments.worlds import SphereWorld
import torch
import pybullet as p
import pybullet_data
import pathlib
import pytorch_kinematics as pk

import time

FLOW_MPC_ROOT = pathlib.Path(__file__).resolve().parents[1]


class KukaSphereWorld(SphereWorld):

    def __init__(self):
        self.obstacle_ids = []
        super().__init__(world_dim=3, world_size=2, min_obstacles=4, max_obstacles=10, min_radius=0.15, max_radius=0.4)

    def _get_occupancy_grid(self):
        obstacle_positions, obstacle_radii = self._generate_environment()
        self._add_obstacles_to_pybullet(obstacle_positions, obstacle_radii)
        occupancy_grid = self._get_occupancy(obstacle_positions, obstacle_radii)
        # add the base to the occupancy grid
        occupancy_grid[24:40, 24:40, :24] = 1.0
        return occupancy_grid

    def _add_obstacles_to_pybullet(self, obstacle_positions, obstacle_radii):
        self.obstacle_ids = []
        for centre, radius in zip(obstacle_positions, obstacle_radii):
            #print(centre, radius)
            v = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius,
                                    rgbaColor=[.7, 0.0, 0.7, 1],
                                    visualFramePosition=[0, 0, 0]
                                    )

            c = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius, collisionFramePosition=[0, 0, 0],
                                       flags=1)

            id = p.createMultiBody(baseMass=0,
                                   baseInertialFramePosition=[0, 0, 0],
                                   baseCollisionShapeIndex=c,
                                   baseVisualShapeIndex=v,
                                   basePosition=[centre[1], centre[0], centre[2]],
                                   useMaximalCoordinates=True)

            self.obstacle_ids.append(id)


class KukaEnv:

    def __init__(self, gui=True, debug=False, dt=0.1, world_type='spheres'):
        self.dt = dt
        self.state_dim = 7
        self.control_dim = 7
        self.gui = gui
        self.sim = None
        self.debug = debug

        self._setup_pybullet()
        self._setup_scene()
        self._setup_camera()
        # self.chain = pk.build_serial_chain_from_urdf(open(f"{FLOW_MPC_ROOT}/models/urdf/kuka_iiwa.urdf").read(),
        #                                             "lbr_iiwa_link_7")

        self.chain = pk.build_serial_chain_from_urdf(
            open(f"{FLOW_MPC_ROOT}/../venv/lib/python3.8/site-packages/pybullet_data/kuka_iiwa/model.urdf").read(),
            "lbr_iiwa_link_7")

        if world_type == 'spheres':
            self.world = KukaSphereWorld()
        self.collision_balls = []

    def _setup_camera(self):
        self.viewMatrix = p.computeViewMatrix(
            cameraEyePosition=[2, 2, 2],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[-1, -1, 1])

        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=10.0)

    def _setup_pybullet(self):
        if self.gui:
            self.sim = p.connect(p.GUI)
        else:
            self.sim = p.connect(p.DIRECT)

        # p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _setup_scene(self):
        # load base & robot arm without additional stuff
        # Base for arm to sit on
        p.loadURDF("plane.urdf", basePosition=[0, 0, -1])
        pos = [0, 0, -1 + 0.375]
        size = [0.25, 0.25, 0.375]

        v = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=size,
                                rgbaColor=[.7, 0.7, 0.7, 1.0],
                                visualFramePosition=pos
                                )

        c = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=size, collisionFramePosition=pos, flags=1
                                   )

        self.base = p.createMultiBody(baseMass=0,
                                      baseInertialFramePosition=[0, 0, 0],
                                      baseCollisionShapeIndex=c,
                                      baseVisualShapeIndex=v,
                                      basePosition=[0, 0, 0],
                                      useMaximalCoordinates=True)

        # add arm to base
        self.robot = p.loadURDF("kuka_iiwa/model.urdf",
                                basePosition=[0., 0., -.25])  # , flags=p.URDF_USE_SELF_COLLISION)

        # add a goal display arm
        self.goal_robot = p.loadURDF("kuka_iiwa/model_visual.urdf",
                                     basePosition=[0., 0., -.25])  # , flags=p.URDF_USE_SELF_COLLISION)

    def step(self, control):
        # commands are joint velocities
        # TODO make the dynamics actually occur in pybullet
        self.state = self.state + self.dt * control
        self._set_state(self.state)
        # p.stepSimulation()
        # ime.sleep(self.dt)
        if self.debug:
            self.remove_collision_balls()

        collision = self.check_collision()

        if self.debug:
            self._display_collision_check_spheres()

            width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                width=224,
                height=224,
                viewMatrix=self.viewMatrix,
                projectionMatrix=self.projectionMatrix)

        return self._get_state(), collision

    def reset(self):
        self.start = None
        self.goal = None
        while self.start is None or self.goal is None:
            print('trying to get start and goal...')
            p.resetSimulation()
            self._setup_scene()
            self.world.reset()

            # attempt to find goal
            # goal = np.pi * (2 * np.random.rand(7) - 1)
            goal = np.random.randn(7)
            self._set_state(goal)
            self.goal = goal if not self.check_collision() else None

            # Attempt to find start
            # start = np.pi * (2 * np.random.rand(7) - 1)
            start = np.random.randn(7)
            self._set_state(start)
            self.start = start if not self.check_collision() else None

        self.state = self.start.copy()
        # self._display_collision_check_spheres()
        for i, theta in enumerate(self.goal):
            p.resetJointState(self.goal_robot, i, theta)

    def _set_state(self, state):
        for i, theta in enumerate(state):
            p.resetJointState(self.robot, i, theta)

    def _get_state(self):
        state = np.zeros(7)
        for i in range(7):
            state[i] = p.getJointState(self.robot, i)[0]

        return state

    def check_collision(self):
        # TODO this can be done in pybullet
        p.stepSimulation()
        contact_points = p.getContactPoints(self.robot)
        for contact_point in contact_points:
            idA, idB = contact_point[1], contact_point[2]
            if self.robot == idA:
                if idB != self.base:
                    self.collision_point = np.array(contact_point[5])
                    return True

            if self.robot == idB:
                if idA != self.base:
                    self.collision_point = np.array(contact_points[6])
                    return True

        return False

    def _display_collision_check_spheres(self):
        r = 0.12

        collision_balls = []
        link_positions = []
        for i in range(7):
            link_positions.append(p.getLinkState(self.robot, i)[0])
        import torch

        theta = torch.from_numpy(self.state)
        link_poses = self.chain.forward_kinematics(theta, end_only=False)
        # link_poses.pop('lbr_iiwa_link_7')
        offsets = torch.tensor([[0.0, 0.0, 0.075],
                                [0.0, 0.0, -0.075]])
        look_up_points = []
        link_points = []
        for link in link_poses.keys():
            if link == 'lbr_iiwa_link_6' or link == 'lbr_iiwa_link_1' or link == 'lbr_iiwa_link_7':
                points = link_poses[link].transform_points(torch.zeros(1, 3, device=offsets.device)).reshape(-1, 3)
            else:
                points = link_poses[link].transform_points(offsets).reshape(-1, 3)

            points = points + torch.tensor([0.0, 0.0, -0.25])
            look_up_points.append(points)
            # link_points.append(link_poses[link].transform_points(torch.zeros(1, 3)))

        look_up_points = torch.cat(look_up_points, dim=0).numpy()

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

        for point in link_points:
            print(point[0].numpy())
            point = point[0].numpy()
            point[2] -= 0.25
            v = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.12,
                                    rgbaColor=[.8, 0.0, 0., 0.7],
                                    visualFramePosition=[0, 0, 0]
                                    )
            print(v)
            c = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=0.12, collisionFramePosition=[0, 0, 0], flags=1)
            b = p.createMultiBody(baseMass=0,
                                  baseInertialFramePosition=[0, 0, 0],
                                  baseVisualShapeIndex=v,
                                  baseCollisionShapeIndex=c,
                                  basePosition=point,
                                  useMaximalCoordinates=True)

        self.collision_balls = collision_balls

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


if __name__ == "__main__":
    env = KukaEnv()
    env.reset()
    env.world.render()
    # env.reset_start_and_goal()
    for _ in range(1000):
        state, collision = env.step(np.random.randn(7))
        if collision:
            print('COLLISION')
            time.sleep(10)
        time.sleep(0.1)
