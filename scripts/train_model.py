import argparse
import json
import os

import ipdb
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from flow_mpc import flows
from torch import autograd
from torch.utils.data import DataLoader, IterableDataset
from flow_mpc import utils
from visualise_flow_training import visualize_flow, visualize_flow_mujoco
from flow_mpc.environments import DoubleIntegratorEnv
from dm_control.composer import Environment
from flow_mpc.environments import NavigationFixObstacle

np.random.seed(0)
torch.manual_seed(0)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=7000)
    parser.add_argument('--print-epochs', type=int, default=10)
    parser.add_argument('--condition-prior', type=bool, default=True)
    parser.add_argument('--horizon', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--flow-length', type=int, default=10)
    # parser.add_argument('--use-true-grad', action='store_true')
    parser.add_argument('--disable-flow', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--multi-gpu', action='store_true')
    # parser.add_argument('--logging', action='store_true')
    # parser.add_argument('--name', type=str, required=True)
    # parser.add_argument('--use-vae', action='store_true')
    parser.add_argument('--data-file', type=str, default='../data/training_traj/disk_2d_fixed_obstacle_no_noise/disk_2d_fixed_obstacle_no_noise.npz', help="training data")
    parser.add_argument('--action-noise', type=float, default=0.0)
    parser.add_argument('--train-val-ratio', type=int, default=0.8)
    parser.add_argument('--flow-type', type=str, choices=['ffjord', 'nvp', 'otflow'], default='nvp')
    parser.add_argument('--dist-metric', type=str, choices=['L2', 'frechet'], default='L2', help="the distance metric between two sets of trajectory")
    parser.add_argument('--name', type=str, default='test3', help="name of this trial")
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
    if not os.path.exists("runs/" + args.name):
        os.makedirs("runs/" + args.name)
    with open("runs/" + args.name + "/args.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return args


class NaiveMLPModel(nn.Module):
    def __init__(self, state_dim, action_dim, horizon, middle_layer=[256, 256]):
        super(NaiveMLPModel, self).__init__()
        mlp = []
        in_node = state_dim + action_dim * horizon
        for out_node in middle_layer:
            mlp.append(nn.Linear(in_node, out_node))
            mlp.append(nn.ReLU())
            in_node = out_node
        mlp.append(nn.Linear(in_node, state_dim * horizon))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Given state at start time and actions along a time horizon, return the predicted state trajectory
        :param state: torch tensor of shape (B, state_dim)
        :param action: torch tensor of shape (B, horizon, action_dim)
        :return predicted_traj: torch tensor of shape (B, horizon, state_dim)
        """
        batch_size = state.shape[0]
        state_dim = state.shape[1]
        context = torch.cat((state, action.reshape(batch_size, -1)), dim=1)
        predicted_traj = self.mlp(context).reshape(batch_size, -1, state_dim)
        return predicted_traj


def eval_model(model, dataloader, args):
    model.eval()
    loss = []
    for data in dataloader:
        # import ipdb
        # ipdb.set_trace()
        start_state = data[0].double().reshape(-1, *data[0].shape[2:]).to(args.device)
        traj = data[1].double().reshape(-1, *data[1].shape[2:])[:, 0:args.horizon, :].to(args.device)
        action = data[2].double().reshape(-1, *data[2].shape[2:])[:, 0:args.horizon, :].to(args.device)
        if isinstance(model, NaiveMLPModel):
            loss_fn = nn.MSELoss()
            loss.append(loss_fn(model(start_state, action), traj))
        elif isinstance(model, flows.RealNVPModel):
            z, log_prob = model(start_state, action, reverse=True, traj=traj)
            loss.append(-log_prob.mean())
    model.train()
    return torch.tensor(loss).mean().item()


if __name__ == "__main__":
    visualize_fn = visualize_flow_mujoco
    args = parse_arguments()
    writer = SummaryWriter(log_dir="runs/" + args.name)
    if args.disable_flow:
        model = NaiveMLPModel(state_dim=4, action_dim=2, horizon=args.horizon).double()
    elif args.flow_type == 'nvp':
        model = flows.RealNVPModel(state_dim=4, action_dim=2, horizon=args.horizon, hidden_dim=args.hidden_dim, condition=args.condition_prior, flow_length=args.flow_length).double()
    else:
        raise NotImplementedError
    model.to(args.device)
    model.train()
    data_list = dict(np.load(args.data_file))
    all_start_state = torch.tensor(data_list['starts']).double().reshape(-1, *data_list['starts'].shape[2:]).to(args.device)
    all_action = torch.tensor(data_list['U']).double().reshape(-1, *data_list['U'].shape[2:]).to(args.device)
    all_traj = torch.tensor(data_list['states']).double().reshape(-1, *data_list['states'].shape[2:]).to(args.device)
    model(all_start_state[:int(0.01*all_start_state.shape[0])], all_action[:int(0.01*all_action.shape[0])], reverse=True, traj=all_traj[:int(0.01*all_traj.shape[0])])     # actnorm layer initialization
    data_list = list(zip(*data_list.values()))
    train_val_split = int(args.train_val_ratio * len(data_list))
    train_dataloader = DataLoader(data_list[0:train_val_split], batch_size=args.batch_size, shuffle=True)
    validate_dataloader = DataLoader(data_list[train_val_split:], batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # env = DoubleIntegratorEnv(world_dim=2, world_type='spheres', dt=0.05, action_noise_cov=args.action_noise*np.eye(2))
    env = Environment(NavigationFixObstacle())
    visualize_fn(env, model)
    for epoch in range(args.epochs):
        # print(f"epoch: {epoch}")
        for data in train_dataloader:
            start_state = data[0].double().reshape(-1, *data[0].shape[2:]).to(args.device)
            traj = data[1].double().reshape(-1, *data[1].shape[2:])[:, 0:args.horizon, :].to(args.device)
            action = data[2].double().reshape(-1, *data[2].shape[2:])[:, 0:args.horizon, :].to(args.device)
            if isinstance(model, NaiveMLPModel):
                loss_fn = nn.MSELoss()
                loss = loss_fn(model(start_state, action), traj)
            elif isinstance(model, flows.RealNVPModel):
                z, log_prob = model(start_state, action, reverse=True, traj=traj)
                loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('epoch/train loss', loss, epoch)
        if (epoch) % args.print_epochs == 0:
            dist, std_true, std_pred, prior_std = visualize_fn(env, model, dist_type=args.dist_metric, title=args.name)
            prior_std = prior_std.mean()
            writer.add_scalar('epoch/trajectory prediction error', dist, epoch)
            writer.add_scalar('epoch/true traj std', std_true, epoch)
            writer.add_scalar('epoch/pred traj std', std_pred, epoch)
            writer.add_scalar('epoch/prior std', prior_std, epoch)
            test_loss = eval_model(model, validate_dataloader, args)
            writer.add_scalar('epoch/test loss', test_loss, epoch)
            print(f"epoch: {epoch} | test loss: {test_loss:.3g} | train loss: {loss:.3g} "
                  + f"| prediction error: {dist:.3g} | true traj std: {std_true} | pred traj std: {std_pred} | prior std: {prior_std}")
    writer.close()
    utils.save_checkpoint(model, optimizer, f"../data/{args.name}.pt")
    ipdb.set_trace()
