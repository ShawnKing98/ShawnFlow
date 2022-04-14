import argparse
import json
import os
import time
import ipdb

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import autograd
from torch.utils.data import DataLoader, IterableDataset
import torch.nn.functional as F
from torch import nn

from flow_mpc import flows
from flow_mpc import utils
from flow_mpc.environments import DoubleIntegratorEnv
from flow_mpc.environments import NavigationObstacle

from dm_control.composer import Environment

from visualise_flow_training import visualize_flow, visualize_flow_mujoco, visualize_flow_from_data
from dataset import TrajectoryDataset, TrajectoryImageDataset

np.random.seed(0)
torch.manual_seed(0)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=2**2)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--print-epochs', type=int, default=20)
    parser.add_argument('--condition-prior', action='store_true')
    parser.add_argument('--with-image', type=bool, default=True)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--flow-length', type=int, default=10)
    # parser.add_argument('--use-true-grad', action='store_true')
    parser.add_argument('--disable-flow', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--multi-gpu', action='store_true')
    # parser.add_argument('--logging', action='store_true')
    # parser.add_argument('--name', type=str, required=True)
    # parser.add_argument('--use-vae', action='store_true')
    parser.add_argument('--data-file', type=str, default='../data/training_traj/mul_start_disk_2d_env_2/mul_start_disk_2d_env_2.npz', help="training data")
    parser.add_argument('--action-noise', type=float, default=0.1)
    parser.add_argument('--process-noise', type=float, default=0.000)
    parser.add_argument('--train-val-ratio', type=int, default=0.95)
    parser.add_argument('--flow-type', type=str, choices=['ffjord', 'nvp', 'otflow'], default='nvp')
    parser.add_argument('--dist-metric', type=str, choices=['L2', 'frechet'], default='L2', help="the distance metric between two sets of trajectory")
    parser.add_argument('--name', type=str, default='test', help="name of this trial")
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

def train_model(model, dataloader, args):
    t0 = time.time()
    tik = t0
    for data in dataloader:
        t1 = time.time()
        if isinstance(model, NaiveMLPModel):
            start_state, traj, action = data
            start_state = start_state.squeeze(1).to(args.device)
            traj = traj.to(args.device)
            action = action.to(args.device)
            t2 = time.time()
            loss_fn = nn.MSELoss()
            loss = loss_fn(model(start_state, action), traj)
        elif isinstance(model, flows.RealNVPModel):
            start_state, traj, action = data
            start_state = start_state.squeeze(1).to(args.device)
            traj = traj.to(args.device)
            action = action.to(args.device)
            t2 = time.time()
            # ipdb.set_trace()
            z, log_prob = model(start_state, action, reverse=True, traj=traj)
            t3 = time.time()
            loss = -log_prob.mean()
        elif isinstance(model, flows.ImageRealNVPModel):
            start_state, traj, action, image = data
            N = start_state.shape[1]
            start_state = start_state.reshape(-1, *start_state.shape[-1:]).to(args.device)
            traj = traj.reshape(-1, *traj.shape[-2:]).to(args.device)
            action = action.reshape(-1, *action.shape[-2:]).to(args.device)
            image = image.unsqueeze(1).repeat(1, N, 1, 1, 1)
            image = image.reshape(-1, *image.shape[-3:]).to(args.device)
            t2 = time.time()
            z, log_prob = model(start_state, action, image, reverse=True, traj=traj)
            t3 = time.time()
            loss = -log_prob.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t4 = time.time()
        tt = np.array([t0, t1, t2, t3, t4])
        dt = tt[1:]-tt[:-1]
        dt = dt/dt.sum()*100
        # print(f"sample batch: {dt[0]:.1f}% | move to cuda: {dt[1]:.1f}% | forward propagation: {dt[2]:.1f}% | back propagation: {dt[3]:.1f}%")
        t0 = time.time()
    tok = time.time()
    # print(f"train one epoch: {tok-tik} sec.")
    return loss.item(), tok-tik

def eval_model(model, dataloader, args):
    model.eval()
    loss = []
    for data in dataloader:
        if isinstance(model, NaiveMLPModel):
            start_state, traj, action = data
            start_state = start_state.squeeze(1).to(args.device)
            traj = traj.to(args.device)
            action = action.to(args.device)
            loss_fn = nn.MSELoss()
            loss.append(loss_fn(model(start_state, action), traj))
        elif isinstance(model, flows.RealNVPModel):
            start_state, traj, action = data
            start_state = start_state.squeeze(1).to(args.device)
            traj = traj.to(args.device)
            action = action.to(args.device)
            with torch.no_grad():
                z, log_prob = model(start_state, action, reverse=True, traj=traj)
                loss.append(-log_prob.mean().item())
        elif isinstance(model, flows.ImageRealNVPModel):
            start_state, traj, action, image = data
            N = start_state.shape[1]
            start_state = start_state.reshape(-1, *start_state.shape[-1:]).to(args.device)
            traj = traj.reshape(-1, *traj.shape[-2:]).to(args.device)
            action = action.reshape(-1, *action.shape[-2:]).to(args.device)
            image = image.unsqueeze(1).repeat(1, N, 1, 1, 1)
            image = image.reshape(-1, *image.shape[-3:]).to(args.device)
            with torch.no_grad():
                z, log_prob = model(start_state, action, image, reverse=True, traj=traj)
                loss.append(-log_prob.mean().item())
    model.train()
    return sum(loss)/len(loss)


if __name__ == "__main__":
    visualize_fn = visualize_flow_mujoco
    args = parse_arguments()
    writer = SummaryWriter(log_dir="runs/" + args.name)
    data_dict = dict(np.load(args.data_file))
    state_mean = data_dict['states'].mean((0, 1, 2))
    state_std = data_dict['states'].std((0, 1, 2))
    action_mean = data_dict['U'].mean((0, 1, 2))
    action_std = data_dict['U'].std((0, 1, 2))
    if 'views' in data_dict.keys():
        image_mean = data_dict['views'].mean()
        image_std = data_dict['views'].std()
    data_list = list(zip(*data_dict.values()))
    train_val_split = int(args.train_val_ratio * len(data_list))
    train_data_tuple = tuple(zip(*data_list[0:train_val_split]))
    val_data_tuple = tuple(zip(*data_list[train_val_split:]))
    train_data_tuple = tuple(np.array(data) for data in train_data_tuple)
    val_data_tuple = tuple(np.array(data) for data in val_data_tuple)
    del data_list
    # train_data_tuple = tuple(data_dict.values())
    # val_data_tuple = tuple(data_dict.values())
    if not args.with_image:
        train_dataloader = DataLoader(TrajectoryDataset(train_data_tuple, args.horizon), batch_size=args.batch_size, shuffle=True, num_workers=8)
        validate_dataloader = DataLoader(TrajectoryDataset(val_data_tuple, args.horizon), batch_size=args.batch_size, shuffle=True, num_workers=8)
    else:
        train_dataloader = DataLoader(TrajectoryImageDataset(train_data_tuple, args.horizon), batch_size=args.batch_size,
                                      shuffle=True, num_workers=8)
        validate_dataloader = DataLoader(TrajectoryImageDataset(val_data_tuple, args.horizon), batch_size=args.batch_size,
                                         shuffle=True, num_workers=8)
    del train_data_tuple
    # del val_data_tuple
    if args.disable_flow:
        model = NaiveMLPModel(state_dim=4, action_dim=2, horizon=args.horizon)
    elif args.flow_type == 'nvp':
        if not args.with_image:
            model = flows.RealNVPModel(state_dim=4, action_dim=2, horizon=args.horizon, hidden_dim=args.hidden_dim, condition=args.condition_prior, flow_length=args.flow_length).float()
        else:
            model = flows.ImageRealNVPModel(state_dim=4, action_dim=2, horizon=args.horizon, image_size=(128, 128),
                                            hidden_dim=args.hidden_dim, condition=args.condition_prior, flow_length=args.flow_length,
                                            state_mean=state_mean, state_std=state_std, action_mean=action_mean, action_std=action_std, image_mean=image_mean, image_std=image_std).float()
    else:
        raise NotImplementedError
    model.to(args.device)
    model.train()
    print(f"The number of the model's trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # env = DoubleIntegratorEnv(world_dim=2, world_type='spheres', dt=0.05, action_noise_cov=args.action_noise*np.eye(2))
    env = Environment(NavigationObstacle(process_noise=args.process_noise, action_noise=args.action_noise), max_reset_attempts=2)
    # visualize_fn(env, model, horizon=args.horizon, dist_type=args.dist_metric, title=args.name)
    best_dist = np.inf
    for epoch in range(args.epochs):
        train_loss, epoch_time = train_model(model, train_dataloader, args)
        writer.add_scalar('epoch/train loss', train_loss, epoch)
        print(f"epoch: {epoch} | loss: {train_loss} | time: {epoch_time:.1f} sec.")
        if epoch % args.print_epochs == 0:
            # dist, std_true, std_pred, prior_std = visualize_fn(env, model, horizon=args.horizon, dist_type=args.dist_metric, title=args.name)
            dist, std_true, std_pred, prior_std = visualize_flow_from_data(data=val_data_tuple, flow=model, device=args.device, dist_type=args.dist_metric, horizon=args.horizon)
            prior_std = prior_std.mean()
            writer.add_scalar('epoch/trajectory prediction error', dist, epoch)
            writer.add_scalar('epoch/true traj std', std_true, epoch)
            writer.add_scalar('epoch/pred traj std', std_pred, epoch)
            writer.add_scalar('epoch/prior log std', prior_std, epoch)
            test_loss = eval_model(model, validate_dataloader, args)
            writer.add_scalar('epoch/test loss', test_loss, epoch)
            utils.save_checkpoint(model, optimizer, f"../data/flow_model/{args.name}/{args.name}.pt")
            print(f"epoch: {epoch} | test loss: {test_loss:.3g} | train loss: {train_loss:.3g} "
                  + f"| prediction error: {dist:.3g} | true traj std: {std_true} | pred traj std: {std_pred} | prior log std: {prior_std}")
            if dist < best_dist:
                best_dist = dist
                utils.save_checkpoint(model, optimizer, f"../data/flow_model/{args.name}/{args.name}_best.pt")
    writer.close()
    # ipdb.set_trace()
