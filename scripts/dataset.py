import ipdb
import torch
import numpy as np
from typing import List, Tuple
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt


class TrajectoryDataset(torch.utils.data.Dataset):
    """
    A custom dataset to sample sub-trajectories from collected trajectories
    """
    def __init__(self,
                 data_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 sub_length: int = None,
                 with_contact: bool = False,
                 device="cpu"
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
        self.full_traj = torch.cat((start_state.unsqueeze(-2), traj), dim=-2).to(device)     # shape (E, N, horizon+1, state_dim)
        self.action = torch.tensor(data_tuple[2], dtype=torch.float, device=device)          # shape (E, N, horizon, action_dim)
        # self.action = self.action.reshape(-1, *self.action.shape[-2:])          # shape (E*N, horizon, action_dim)
        self.sub_length = sub_length if sub_length is not None else traj.shape[-2]
        print(f"The shape of full trajectory: {self.full_traj.shape}")
        print(f"The shape of action: {self.action.shape}")
        print(f"The length of sub-trajectory: {self.sub_length}")
        if with_contact:
            self.contact_flag = torch.tensor(data_tuple[4], dtype=torch.float, device=device)   # shape(E, N, horizon)
            print(f"The shape of contact flag: {self.contact_flag.shape}")
        else:
            self.contact_flag = None

    def __len__(self):
        return len(self.full_traj)

    def __getitem__(self, idx):
        # ipdb.set_trace()
        # start_idx = np.random.randint(0, self.full_traj.shape[-2] - self.sub_length)
        # start_idx = random.randint(0, self.full_traj.shape[-2] - self.sub_length - 1)
        # start_idx = 0
        start_idx = torch.randint(self.full_traj.shape[-2] - self.sub_length, (self.full_traj.shape[1],))       # shape(N,)
        idx_sequence = (start_idx.unsqueeze(-1) + torch.arange(self.sub_length).repeat(self.full_traj.shape[1], 1)).unsqueeze(-1)        # shape (N, sub_length, 1)
        traj_idx = (idx_sequence+1).repeat(1, 1, self.full_traj.shape[-1])     # shape (N, sub_length, state_dim)
        action_idx = idx_sequence.repeat(1, 1, self.action.shape[-1])      # shape (N, sub_length, action_dim)
        # start_state = self.full_traj[idx, :, start_idx]         # shape (N, state_dim)
        # traj = self.full_traj[idx, :, start_idx+1: start_idx+self.sub_length+1]     # shape (N, sub_length, state_dim)
        # action = self.action[idx, :, start_idx: start_idx+self.sub_length]          # shape (N, sub_length, action_dim)
        start_state = self.full_traj[idx, torch.arange(self.full_traj.shape[1]), start_idx]  # shape (N, state_dim)
        traj = self.full_traj[idx].gather(dim=1, index=traj_idx)    # shape (N, sub_length, state_dim)
        action = self.action[idx].gather(dim=1, index=action_idx)   # shape (N, sub_length, action_dim)
        if self.contact_flag is None:
            return start_state, traj, action
        else:
            contact_flag = self.contact_flag[idx].gather(dim=1, index=idx_sequence.squeeze(-1))   # shape (N, sub_length)
            return start_state, traj, action, contact_flag
    

class TrajectoryImageDataset(TrajectoryDataset):
    """
    A custom dataset to sample sub-trajectories from collected trajectories, along with the image of the environment
    """
    def __init__(self,
                 data_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 sub_length: int = None,
                 with_contact: bool = False,
                 device="cpu"):
        super(TrajectoryImageDataset, self).__init__(data_tuple, sub_length, with_contact=with_contact, device=device)
        self.sample_per_env = data_tuple[0].shape[1]
        self.image = torch.tensor(data_tuple[3], dtype=torch.float, device=device)     # shape (E, channel, height, width)
        if self.image.dim() < 4:
            self.image.unsqueeze_(1)        # create channel dimension if needed

    def __getitem__(self, idx):
        items = super(TrajectoryImageDataset, self).__getitem__(idx)
        # image = self.image[idx//self.sample_per_env]
        image = self.image[idx]     # shape (channel, height, width)

        return *items, image

if __name__ == "__main__":
    data_tuple = dict(np.load('../data/training_traj/full_disk_2d_with_contact_env_1/full_disk_2d_with_contact_env_1.npz'))
    data_tuple = tuple(data_tuple.values())
    S = TrajectoryDataset(data_tuple, 2)
    S = TrajectoryImageDataset(data_tuple, 20, with_contact=True)
    loader = DataLoader(S, batch_size=512, shuffle=True)
    ii = S[23][-1][0]
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


