import argparse
import json
import os
import time
import ipdb

import numpy as np
import torch
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from torch import autograd
from torch.utils.data import DataLoader, IterableDataset
import torch.nn.functional as F
from torch import nn

from flow_mpc import flows
from flow_mpc import utils
# from flow_mpc.environments import DoubleIntegratorEnv
# from flow_mpc.environments import NavigationObstacle

# from dm_control.composer import Environment

from visualise_flow_training import visualize_flow, visualize_flow_mujoco, visualize_flow_from_data
from visualize_rope_2d_training import visualize_rope_2d_from_data
from dataset import TrajectoryDataset, TrajectoryImageDataset

np.random.seed(0)
torch.manual_seed(0)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=2**8)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--print-epochs', type=int, default=20)
    parser.add_argument('--condition-prior', type=bool, default=True)
    parser.add_argument('--with-image', type=bool, default=True)
    parser.add_argument('--state-dim', type=int, default=4)
    parser.add_argument('--control-dim', type=int, default=2)
    parser.add_argument('--env-dim', type=int, default=64)
    parser.add_argument('--double-flow', type=bool, default=False, help="whether to enable the double flow architecture")
    parser.add_argument('--with-contact', type=bool, default=True)
    parser.add_argument('--pre-rotation', type=bool, default=False)
    parser.add_argument('--contact-dim', type=int, default=None, help="the contact status dimension at one timestamp")
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--flow-length', type=int, default=10)
    # parser.add_argument('--use-true-grad', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--multi-gpu', action='store_true')
    # parser.add_argument('--logging', action='store_true')
    # parser.add_argument('--name', type=str, required=True)
    # parser.add_argument('--use-vae', action='store_true')
    parser.add_argument('--data-file', type=str, default="../data/training_traj/full_disk_2d_with_contact_env_1/full_disk_2d_with_contact_env_1.npz", help="training data")
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--last-epoch', type=int, default=0)
    parser.add_argument('--action-noise', type=float, default=0.1)
    parser.add_argument('--process-noise', type=float, default=0.000)
    parser.add_argument('--train-val-ratio', type=float, default=0.95)
    parser.add_argument('--flow-type', type=str, choices=['ffjord', 'nvp', 'otflow', 'autoregressive', 'msar'], default='autoregressive')
    parser.add_argument('--dist-metric', type=str, choices=['L2', 'frechet'], default='L2', help="the distance metric between two sets of trajectory")
    parser.add_argument('--name', type=str, default='disk_autoregressive_13_retrain', help="name of this trial")
    parser.add_argument('--remark', type=str, default='autoregressive 13 retrain', help="any additional information")
    # parser.add_argument('--vae-flow-prior', action='store_true')
    # parser.add_argument('--supervised', action='store_true')
    # parser.add_argument('--load-vae', type=str, default=None)
    # parser.add_argument('--vae-training-epochs', type=int, default=100)
    # parser.add_argument('--env', type=str,
    #                     choices=['double_integrator_2d', 'double_integrator_3d', 'quadcopter_2d', 'quadcopter_3d',
    #                              'quadcopter_2d_dynamic', 'quadcopter_3d_dynamic',
    #                              'dubins_car', 'victor_tabletop'],
    #                     default='double_integrator_2d')

    args = parser.parse_args()
    for (arg, value) in args._get_kwargs():
        print(f"{arg}: {value}")
    if not os.path.exists("../data/flow_model/" + args.name):
        os.makedirs("../data/flow_model/" + args.name)
    with open("../data/flow_model/" + args.name + "/args.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return args

if __name__ == "__main__":
    args = parse_arguments()
