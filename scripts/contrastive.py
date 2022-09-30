import torch
from torch import nn
from flow_mpc import flows
from flow_mpc.encoders import Encoder

class Aligner(nn.Module):
    def __init__(self, feature_dim, image_size, state_dim, horizon,
                state_mean=None, state_std=None, image_mean=None, image_std=None):
        super(Aligner, self).__init__()
        self.nonlinearity = nn.ReLU()
        self.env_encoder = Encoder(image_size, feature_dim)
        self.traj_encoder = nn.Sequential(
            nn.Linear(state_dim*horizon, 256),
            self.nonlinearity,
            nn.Linear(256, 256),
            self.nonlinearity,
            nn.Linear(256, feature_dim)
        )
        self.register_buffer('state_mean', torch.tensor(state_mean, dtype=torch.float) if state_mean is not None else torch.zeros(state_dim))
        self.register_buffer('state_std', torch.tensor(state_std, dtype=torch.float) if state_std is not None else torch.ones(state_dim))
        self.register_buffer('image_mean', torch.tensor(image_mean, dtype=torch.float) if image_mean is not None else torch.zeros(()))
        self.register_buffer('image_std', torch.tensor(image_std, dtype=torch.float) if image_std is not None else torch.ones(()))
    def forward(self, image: torch.Tensor, traj: torch.Tensor, cross_align=True):
        """
        Given a set of images and a set of trajectories, output the alignment scores
        :param image: a tensor of shape (B_image, C, H, W)
        :param traj: a tensor of shape (B_traj, horizon, state_dim)
        :param cross_align: whether to output an alignment matrix. If not, B_traj has to be able to be divided by B_image
        :return alignment: a tensor of shape (B_traj, B_image) if cross_align is True or a tensor of shape (B_traj,)
        """
        image = (image - self.image_mean) / self.image_std
        traj = (traj - self.state_mean) / self.state_std
        B_traj = traj.shape[0]
        image_feature = self.env_encoder.encode(image)
        traj_feature = self.traj_encoder(traj.reshape(B_traj, -1))
        image_feature = image_feature / image_feature.norm(dim=1, keepdim=True)
        traj_feature = traj_feature / traj_feature.norm(dim=1, keepdim=True)
        if not cross_align:
            # assert image.shape[0] == traj.shape[0]
            N = traj.shape[0] // image_feature.shape[0]
            image_feature = image_feature.unsqueeze(0).repeat(N, 1, 1).transpose(0, 1).reshape(-1, image_feature.shape[1])
            alignment = (image_feature * traj_feature).sum(1)
        else:
            alignment = traj_feature @ image_feature.T
        return alignment

if __name__ == "__main__":
    model = Aligner(64, (128, 128), 4, 10)
