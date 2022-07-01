import numpy as np
import argparse
import json

import dm_control.composer.environment
import ipdb
import matplotlib.figure
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import copy

from dm_control.composer import Environment
from flow_mpc.environments import NavigationFixObstacle
from flow_mpc.environments import DoubleIntegratorEnv
from flow_mpc.controllers import RandomController
from flow_mpc.flows import RealNVPModel, ImageRealNVPModel, ConditionalPrior, GaussianPrior
from flow_mpc import utils
from frechetdist import frdist

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-num', type=int, default=50)
    parser.add_argument('--flow-name', type=str, default="disk_2d_image_flow")
    parser.add_argument('--data-name', type=str, default="disk_2d_variable_obstacle_noisy")
    parser.add_argument('--world-dim', type=int, default=2)
    parser.add_argument('--control-dim', type=int, default=2)
    parser.add_argument('--state-dim', type=int, default=4)
    parser.add_argument('--with-image', type=bool, default=False)
    args = parser.parse_args()
    args.flow_path = f"../data/flow_model/{args.flow_name}_best.pt"
    with open(f"./runs/{args.flow_name}/args.txt") as f:
        stored_args = f.read()
    stored_args = json.loads(stored_args)
    for (k, v) in stored_args.items():
        # if getattr(args, k, None) is None:
        setattr(args, k, v)
    args.dist_metric = 'frechet'
    for (arg, value) in args._get_kwargs():
        print(f"{arg}: {value}")
    # ipdb.set_trace()
    return args

if __name__ == '__main__':
    args = parse_args()
    data_dict = dict(np.load(f"../data/training_traj/{args.data_name}/{args.data_name}.npz"))
    if not args.with_image:
        model = RealNVPModel(state_dim=args.state_dim, action_dim=args.control_dim, horizon=args.horizon, hidden_dim=args.hidden_dim, condition=args.condition_prior, flow_length=args.flow_length, initialized=True).float().to(args.device)
    else:
        model = ImageRealNVPModel(state_dim=args.state_dim, action_dim=args.control_dim, horizon=args.horizon, image_size=(128, 128), hidden_dim=args.hidden_dim, condition=args.condition_prior, flow_length=args.flow_length, initialized=True).float().to(args.device)
    utils.load_checkpoint(model, filename=args.flow_path)
    model.eval()
