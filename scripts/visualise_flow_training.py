import pathlib
import argparse
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from flow_mpc.environments import DoubleIntegratorEnv
from flow_mpc.controllers import RandomController
from flow_mpc.flows import RealNVPModel, ConditionalPrior, GaussianPrior
from flow_mpc import utils
from frechetdist import frdist

np.random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--flow-path', type=str, default="../data/0_action_noise_3_layer_5e-4_lr.pth")
    parser.add_argument('--condition-prior', type=bool, default=True)
    parser.add_argument('--action-noise', type=float, default=0.0)
    parser.add_argument('--world-dim', type=int, default=2)
    parser.add_argument('--control-dim', type=int, default=2)
    parser.add_argument('--state-dim', type=int, default=4)
    parser.add_argument('--horizon', type=int, default=40, help="The length of the future state trajectory to be considered")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return args


def visualize_flow(env, flow: nn.Module, controller=None, device="cuda" if torch.cuda.is_available() else "cpu", dist_type: str=None, title=None):
    """
    Visualize the predicted trajectory distribution vs. ground truth trajectory distribution,
    can also return some stochastic metric of the two sets of trajectories
    :param env: an env instance
    :param flow: a flow model
    :param controller: a controller
    :param device: running device
    :param dist_type: the type of distance metric, choice of [L2, frechet, None], None stands for no calculating at all
    :return dist: the average distance between two sets of trajectories
    :return std_true: the std among the ground truth trajectories
    :return std_pred: the std among the predicted trajectories
    """
    flow.eval()
    if controller is None:
        controller = RandomController([-2] * env.control_dim, [2] * env.control_dim, env.control_dim, 40)
    env.reset()
    while not env.reset_start_and_goal():
        continue
    control_sequence = controller.step(env.start)
    start_tensor = torch.tensor(env.start, device=device, dtype=torch.double).unsqueeze(0)
    action = torch.tensor(control_sequence, device=device, dtype=torch.double).unsqueeze(0)

    # calculate the variance of the prior
    prior = flow.prior
    if isinstance(prior, GaussianPrior):
        prior_std = prior.std
    elif isinstance(prior, ConditionalPrior):
        context = torch.cat((start_tensor, action.reshape(1, -1)), dim=1)
        prior_mu, prior_std = torch.chunk(prior.fc2(prior.act_fn(prior.fc1(context))), chunks=2, dim=1)
        prior_std = torch.sigmoid(prior_std) + 1e-7
    else:
        raise Exception("Unknown prior type")

    pred_traj_history = torch.zeros((0, len(control_sequence), env.state_dim), device=device)
    true_traj_history = torch.zeros((0, len(control_sequence), env.state_dim), device=device)
    ldj_history = torch.zeros(0, device=device)

    plt.clf()
    # draw predicted trajectory
    for i in range(100):
        pred_traj = flow(start_tensor, action)[0]
        pred_traj_history = torch.cat((pred_traj_history, pred_traj), dim=0)
        pred_traj = pred_traj.squeeze(0).cpu().detach().numpy()
        pred_traj = np.concatenate([np.expand_dims(env.start, 0), pred_traj])
        label = 'predicted distribution' if i == 0 else None
        plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'c', label=label, linewidth=4, alpha=0.1)

    # draw ground truth trajectory
    for i in range(100):
        env.state = env.start
        single_traj = [env.start]
        for t in range(len(control_sequence)):
            env.step(control_sequence[t])
            single_traj.append(env.state)
        single_traj = np.stack(single_traj, axis=0)
        true_traj = torch.tensor(single_traj[1:, :], device=device, dtype=pred_traj_history.dtype).unsqueeze(0)
        true_traj_history = torch.cat((true_traj_history, true_traj), dim=0)
        label = 'ground truth distribution' if i == 0 else None
        plt.plot(single_traj[:, 0], single_traj[:, 1], 'r', label=label, linewidth=1, alpha=0.5)
    plt.plot(env.start[0], env.start[1], 'bo', markersize=10)
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.pause(0.2)
    plt.show()

    # calculate distance between two sets of trajectories
    if dist_type is None:
        dist = None
    elif dist_type == "frechet":
        # Frechet distance
        pred_traj_history = pred_traj_history.cpu().detach().numpy()
        true_traj_history = true_traj_history.cpu().detach().numpy()
        dist = np.zeros((pred_traj_history.shape[0], true_traj_history.shape[0]))
        for i in range(dist.shape[0]):
            for j in range(dist.shape[1]):
                dist[i, j] = frdist(pred_traj_history[i], true_traj_history[j])
        dist = dist.mean()
    elif dist_type == "L2":
        # L2 distance
        pred_traj_history = pred_traj_history.reshape(pred_traj_history.shape[0], -1)
        true_traj_history = true_traj_history.reshape(true_traj_history.shape[0], -1)
        dist = torch.cdist(pred_traj_history, true_traj_history, p=2).mean()
    else:
        raise Exception(f"No such distance metric type: {dist_type}")
    std_true = true_traj_history.std(dim=0).mean()
    std_pred = pred_traj_history.std(dim=0).mean()
    flow.train()
    return dist, std_true, std_pred, prior_std

if __name__ == "__main__":
    args = parse_args()
    import time
    # environment
    env = DoubleIntegratorEnv(world_dim=args.world_dim, world_type='spheres', dt=0.05, action_noise_cov=args.action_noise*np.eye(args.world_dim))
    # random controller
    controller = RandomController([-2] * args.world_dim, [2] * args.world_dim, args.world_dim, args.horizon)
    # flow model
    model = RealNVPModel(state_dim=args.state_dim, action_dim=args.control_dim, horizon=args.horizon, condition=args.condition_prior).double().to(args.device)
    utils.load_checkpoint(model, filename=args.flow_path)
    model.eval()
    for i in range(100):
        dist, std_true, std_pred, prior_std = visualize_flow(env, model, dist_type='L2', title=args.flow_path)
        time.sleep(1)
        print(f"distance: {dist} | true traj std: {std_true} | pred traj std: {std_pred} | prior std: {prior_std.mean()}")
        # input("?")
    print("yes")
