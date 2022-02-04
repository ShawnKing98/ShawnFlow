import torch
from torch import nn
from flow_mpc.models.generative_model import GenerativeModel
import pytorch_kinematics as pk
import pathlib
from flow_mpc.models.utils import CollisionFcn

FLOW_MPC_ROOT = pathlib.Path(__file__).resolve().parents[1]


class KukaCollisionChecker(nn.Module):

    def __init__(self):
        super().__init__()
        # self.chain = pk.build_serial_chain_from_urdf(open(f"{FLOW_MPC_ROOT}/models/urdf/kuka_iiwa.urdf").read(),
        #                                             "lbr_iiwa_link_7")
        self.chain = pk.build_serial_chain_from_urdf(
            open(f"{FLOW_MPC_ROOT}/../venv/lib/python3.8/site-packages/pybullet_data/kuka_iiwa/model.urdf").read(),
            "lbr_iiwa_link_7")

        self.register_buffer('offsets', torch.tensor([[0.0, 0.0, 0.075],
                                                      [0.0, 0.0, -0.075]]
                                                     ))
        self.register_buffer('base_position', torch.tensor([0.0, 0.0, -0.25]))
        self.ball_radius = 0.12
        # for looking up SDF values batched
        self.sdf_lookup = CollisionFcn()

    def to(self, device):
        self.chain = self.chain.to(device=device)
        return super().to(device=device)

    def apply(self, state, sdf, sdf_grad):
        B, N, dx = state.shape
        link_poses = self.chain.forward_kinematics(state.reshape(B * N, dx), end_only=False)
        # TODO -- check on distance between non-consecutive balls for self collision
        look_up_points = []
        for link in link_poses.keys():
            if link == 'lbr_iiwa_link_6' or link == 'lbr_iiwa_link_1' or link == 'lbr_iiwa_link_7':
                points = link_poses[link].transform_points(torch.zeros(1, 3, device=self.offsets.device)).reshape(-1, 1,
                                                                                                                  3)
            else:
                points = link_poses[link].transform_points(self.offsets).reshape(-1, 2, 3)
            look_up_points.append(points)

        look_up_points = torch.cat(look_up_points, dim=1) + self.base_position
        look_up_points = look_up_points.reshape(B, N, 11, 3)

        # in sdf lookups env is [-2, 2] but for this it is [-1, 1], should make this less hacky
        sdf_lookups = self.sdf_lookup.apply(2 * look_up_points.reshape(B, N * 11, 3), sdf, sdf_grad).reshape(B, N, 11)

        collision = torch.where(sdf_lookups < self.ball_radius, sdf_lookups - self.ball_radius,
                                torch.zeros_like(sdf_lookups))

        closest_point = torch.min(collision, dim=-1).values

        return -1e4 * torch.where(closest_point < 0,
                                  torch.ones_like(closest_point),
                                  torch.zeros_like(closest_point))


class KukaKinematics(nn.Module):
    """
        Kinematic Kuka -- 'dynamics' are simple changes in position
    """

    def __init__(self, dt):
        self.dt = dt
        super().__init__()

    def forward(self, state, control):
        return state + control * self.dt


class KukaModel(GenerativeModel):

    def __init__(self, dt):
        action_prior = torch.distributions.Normal(loc=0.0, scale=1.0)
        super().__init__(dynamics=KukaKinematics(dt), action_prior=action_prior, state_dim=7, control_dim=7)
        self.collision_fn = KukaCollisionChecker()

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
    collision_fn = KukaCollisionChecker().to(device=device)
    th_batch = torch.rand(10, 100, 7, device=device)
    rval = collision_fn.apply(th_batch, torch.zeros(10, 64, 64, 64, device=device), None)
    print(rval.shape)
