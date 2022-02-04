import torch
from torch import nn
from flow_mpc.models.generative_model import GenerativeModel
import pytorch_kinematics as pk
import pathlib
from flow_mpc.models.utils import CollisionFcn

FLOW_MPC_ROOT = pathlib.Path(__file__).resolve().parents[1]

VICTOR_URDF_FILE = '/home/thomas/noetic_catkin_ws/src/kuka_iiwa_interface/victor_description/urdf/victor.urdf'
# VICTOR_URDF_FILE = 'home/tpower/catkin_ws/src/kuka_iiwa_interface/victor_description/urdf/victor.urdf'

class VictorCollisionChecker(nn.Module):

    def __init__(self):
        super().__init__()
        # self.chain = pk.build_serial_chain_from_urdf(open(f"{FLOW_MPC_ROOT}/models/urdf/kuka_iiwa.urdf").read(),
        #                                             "lbr_iiwa_link_7")
        self.chain = pk.build_serial_chain_from_urdf(
            open(VICTOR_URDF_FILE).read(),
            'victor_left_gripper_palm')

        # self.register_buffer('offsets', torch.tensor([[0.0, 0.0, 0.075],
        #                                             [0.0, 0.0, -0.075]]
        #                                             ))
        self.register_buffer('robot_base_position', torch.tensor([-0.65, -0.25, -1]))
        self.register_buffer('ignore_self_collision_flags', torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0],
                                                                          [1, 1, 1, 0, 0, 0, 0, 0],
                                                                          [0, 1, 1, 1, 0, 0, 0, 0],
                                                                          [0, 0, 1, 1, 1, 0, 0, 0],
                                                                          [0, 0, 0, 1, 1, 1, 1, 0],
                                                                          [0, 0, 0, 0, 1, 1, 1, 0],
                                                                          [0, 0, 0, 0, 1, 1, 1, 1],
                                                                          [0, 0, 0, 0, 0, 0, 1, 1]])
                             )
        self.ball_radius = 0.12
        # for looking up SDF values batched
        self.sdf_lookup = CollisionFcn()

    def to(self, device):
        self.chain = self.chain.to(device=device)
        return super().to(device=device)

    def _get_collision_check_points(self, state):
        """
            Takes N x 7 configurations, returns N x num_points x 3 number of sphere centres to collision check
        """

        N, dx = state.shape
        link_poses = self.chain.forward_kinematics(state, end_only=False)
        look_up_points = []

        for link in link_poses.keys():
            if 'victor_left_arm_link' in link and '0' not in link:
                points = link_poses[link].transform_points(torch.zeros(1, 3, device=state.device)).reshape(N, 3)
            elif 'gripper_palm' in link:
                points = link_poses[link].transform_points(
                    torch.tensor([[0.0, 0.0, 0.1]], device=state.device)
                ).reshape(N, 3)
            else:
                continue

            points = points + self.robot_base_position
            look_up_points.append(points)

        return torch.cat(look_up_points, dim=1)

    def apply(self, state, sdf, sdf_grad):
        B, N, dx = state.shape

        look_up_points = self._get_collision_check_points(state.reshape(B * N, dx)).reshape(B, N, -1, 3)
        num_lookups = look_up_points.shape[2]

        # check for self collision
        sq_distance = torch.cdist(look_up_points.reshape(B * N, -1, 3),
                                  look_up_points.reshape(B * N, -1, 3)).reshape(B, N, num_lookups, num_lookups)

        sq_distance = sq_distance + self.ignore_self_collision_flags.reshape(1, 1, num_lookups, num_lookups)
        # Look up in SDF
        sdf_lookups = self.sdf_lookup.forward(None, 4 * look_up_points.reshape(B, -1, 3),
                                              sdf, sdf_grad, check_bounds=False).reshape(B, N, -1)

        closest_points = torch.where(sdf_lookups < self.ball_radius,
                                     sdf_lookups - self.ball_radius,
                                     torch.zeros_like(sdf_lookups))

        closest_point = torch.min(closest_points, dim=-1).values

        collision = torch.where(closest_point < 0,
                                torch.ones_like(closest_point),
                                torch.zeros_like(closest_point))

        self_collision = torch.where(torch.min(sq_distance[:, :, :, 0], dim=-1).values < 2 * self.ball_radius,
                                     torch.ones_like(collision), torch.zeros_like(collision))

        collision = torch.where(self_collision > 0, torch.ones_like(collision), collision)

        return -1e4 * collision


class VictorKinematics(nn.Module):
    """
        Kinematic Kuka -- 'dynamics' are simple changes in position
    """

    def __init__(self, dt):
        self.dt = dt
        super().__init__()

    def forward(self, state, control):
        return state + control * self.dt


class VictorModel(GenerativeModel):

    def __init__(self, dt):
        action_prior = torch.distributions.Normal(loc=0.0, scale=2.0)
        super().__init__(dynamics=VictorKinematics(dt), action_prior=action_prior, state_dim=7, control_dim=7)
        self.collision_fn = VictorCollisionChecker()

    def collision_log_likelihood(self, state, sdf, sdf_grad):
        return self.collision_fn.apply(state, sdf, sdf_grad)

    def goal_log_likelihood(self, state, goal):
        gll = -10 * torch.norm(state - goal, dim=-1)
        return gll

    def to(self, device):
        self.collision_fn = self.collision_fn.to(device=device)
        return super().to(device=device)


if __name__ == '__main__':
    device = 'cuda:0'
    collision_fn = VictorCollisionChecker().to(device=device)
    th_batch = torch.rand(10, 100, 7, device=device)
    rval = collision_fn.apply(th_batch, torch.zeros(10, 64, 64, 64, device=device), None)
    print(rval.shape)
