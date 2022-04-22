import ipdb
import torch
import numpy as np
from typing import List, Tuple
from torch.utils.data import DataLoader
import random


class TrajectoryDataset(torch.utils.data.Dataset):
    """
    A custom dataset to sample sub-trajectories from collected trajectories
    """
    def __init__(self,
                 data_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 sub_length: int = None,
                 device="cuda" if torch.cuda.is_available() else "cpu"
                 ):
        """
        Initialization
        :param data_tuple: a tuple containing start states, action sequences and state trajectories
        :param sub_length: the length of sub-trajectory to be sampled. If none, then the whole trajectory will be returned
        """
        start_state = torch.tensor(data_tuple[0], dtype=torch.float)
        # start_state = start_state.reshape(-1, *start_state.shape[-1:])
        traj = torch.tensor(data_tuple[1], dtype=torch.float)
        # traj = traj.reshape(-1, *traj.shape[-2:])
        self.full_traj = torch.cat((start_state.unsqueeze(-2), traj), dim=-2)     # shape (E, N, horizon+1, state_dim)
        self.action = torch.tensor(data_tuple[2], dtype=torch.float)
        # self.action = self.action.reshape(-1, *self.action.shape[-2:])          # shape (E, N, horizon, action_dim)
        self.sub_length = sub_length if sub_length is not None else traj.shape[-2]
        print(f"The shape of full trajectory: {self.full_traj.shape}")
        print(f"The shape of action: {self.action.shape}")
        print(f"The length of sub-trajectory: {self.sub_length}")

    def __len__(self):
        return len(self.full_traj)

    def __getitem__(self, idx):
        # ipdb.set_trace()
        # start_idx = np.random.randint(0, self.full_traj.shape[-2] - self.sub_length)
        # start_idx = random.randint(0, self.full_traj.shape[-2] - self.sub_length - 1)
        # start_idx = 0
        start_idx = torch.randint(self.full_traj.shape[-2] - self.sub_length, (self.full_traj.shape[1],))       # shape(N,)
        start_idx_tmp = (start_idx.unsqueeze(-1) + torch.arange(self.sub_length).repeat(self.full_traj.shape[1], 1)).unsqueeze(-1)        # shape (N, horizon, 1)
        traj_idx = (start_idx_tmp+1).repeat(1, 1, self.full_traj.shape[-1])     # shape (N, horizon, state_dim)
        action_idx = start_idx_tmp.repeat(1, 1, self.action.shape[-1])      # shape (N, horizon, action_dim)
        # start_state = self.full_traj[idx, :, start_idx]         # shape (N, state_dim)
        # traj = self.full_traj[idx, :, start_idx+1: start_idx+self.sub_length+1]     # shape (N, horizon, state_dim)
        # action = self.action[idx, :, start_idx: start_idx+self.sub_length]          # shape (N, horizon, action_dim)
        start_state = self.full_traj[idx, torch.arange(self.full_traj.shape[1]), start_idx]  # shape (N, state_dim)
        traj = self.full_traj[idx].gather(dim=1, index=traj_idx)    # shape (N, horizon, state_dim)
        action = self.action[idx].gather(dim=1, index=action_idx)   # shape (N, horizon, action_dim)
        return start_state, traj, action
    

class TrajectoryImageDataset(TrajectoryDataset):
    """
    A custom dataset to sample sub-trajectories from collected trajectories, along with the image of the environment
    """
    def __init__(self,
                 data_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 sub_length: int = None,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        super(TrajectoryImageDataset, self).__init__(data_tuple, sub_length, device)
        self.sample_per_env = data_tuple[0].shape[1]
        self.image = torch.tensor(data_tuple[3], dtype=torch.float)     # shape (E, channel, height, width)
        if self.image.dim() < 4:
            self.image.unsqueeze_(1)        # create channel dimension if needed

    def __getitem__(self, idx):
        start_state, traj, action = super(TrajectoryImageDataset, self).__getitem__(idx)
        # image = self.image[idx//self.sample_per_env]
        image = self.image[idx]     # shape (channel, height, width)
        return start_state, traj, action, image

if __name__ == "__main__":
    data_tuple = dict(np.load('../data/training_traj/disk_2d_variable_obstacle_noisy/disk_2d_variable_obstacle_noisy.npz'))
    data_tuple = tuple(data_tuple.values())
    S = TrajectoryDataset(data_tuple, 2)
    S = TrajectoryImageDataset(data_tuple, 20)
    loader = DataLoader(S, batch_size=512, shuffle=True)
    print(S[23])
    # del data_tuple
    # import time
    # l = []
    # for i in range(5):
    #     tik = time.time()
    #     for data in loader:
    #         start_state, traj, action = data
    #         # print(start_state.shape)
    #         # print(traj.shape)
    #         # print(action.shape)
    #         pass
    #     tok = time.time()
    #     l.append(tok-tik)
    # print(sum(l)/len(l))


