import pathlib
import argparse

import dm_control.composer.environment
import ipdb
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dm_control.composer import Environment
from flow_mpc.environments import NavigationFixObstacle
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
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--flow-path', type=str, default="../data/mujoco.pt")
    parser.add_argument('--condition-prior', type=bool, default=True)
    parser.add_argument('--action-noise', type=float, default=0.0)
    parser.add_argument('--world-dim', type=int, default=2)
    parser.add_argument('--control-dim', type=int, default=2)
    parser.add_argument('--state-dim', type=int, default=4)
    parser.add_argument('--horizon', type=int, default=40, help="The length of the future state trajectory to be considered")
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--flow-length', type=int, default=10)
    parser.add_argument('--dist-metric', type=str, choices=['L2', 'frechet'], default='L2',
                        help="the distance metric between two sets of trajectory")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return args


def visualize_flow_mujoco(env: dm_control.composer.environment.Environment,
                          flow: nn.Module,
                          controller=None,
                          device="cuda" if torch.cuda.is_available() else "cpu",
                          dist_type: str=None,
                          test_num: int=100,
                          title=None):
    """
    Visualize the predicted trajectory distribution vs. ground truth trajectory distribution,
    can also return some stochastic metric of the two sets of trajectories
    :param env: a mujoco env instance
    :param flow: a flow model
    :param controller: a controller
    :param device: running device
    :param dist_type: the type of distance metric, choice of [L2, frechet, None], None stands for no calculating at all
    :return dist: the average distance between two sets of trajectories
    :return std_true: the std among the ground truth trajectories
    :return std_pred: the std among the predicted trajectories
    """
    flow.eval()
    control_dim = env.action_spec().shape[0]
    if controller is None:
        # controller = RandomController([-2] * env.control_dim, [2] * env.control_dim, env.control_dim, 40)
        controller = RandomController(udim=control_dim, urange=10, horizon=40, lower_bound=[-10, -10], upper_bound=[10, 10])
    obs = env.reset()
    start_position = obs.observation['robot_position'][0][0:2]
    start_velocity = obs.observation['robot_velocity'][0][0:2]
    start_state = np.concatenate((start_position, start_velocity))
    control_sequence = controller.step(None)
    start_tensor = torch.tensor(start_state, device=device, dtype=torch.double).unsqueeze(0)
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

    pred_traj_history = torch.zeros((0, len(control_sequence), len(start_state)), device=device)
    true_traj_history = torch.zeros((0, len(control_sequence), len(start_state)), device=device)
    ldj_history = torch.zeros(0, device=device)

    plt.clf()
    plt.axis('equal')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    ax = plt.gca()
    wall = patches.Rectangle((-0.71, -0.71), 1.42, 1.42, linewidth=2, edgecolor='k', facecolor='none')
    obs1 = plt.Circle((0, 0.2), 0.1, color='k')
    obs2 = plt.Circle((-0.3, -0.3), 0.1, color='k')
    obs3 = plt.Circle((0.3, -0.3), 0.1, color='k')
    ax.add_patch(wall)
    ax.add_patch(obs1)
    ax.add_patch(obs2)
    ax.add_patch(obs3)
    # plt.plot([-0.71, -0.71, 0.71, 0.71, -0.71], [-0.71, 0.71, 0.71, -0.71, -0.71], 'k', linewidth=2)
    # draw predicted trajectory
    for i in range(test_num):
        # ipdb.set_trace()
        pred_traj = flow(start_tensor, action)[0]
        pred_traj_history = torch.cat((pred_traj_history, pred_traj), dim=0)
        pred_traj = pred_traj.squeeze(0).cpu().detach().numpy()
        pred_traj = np.concatenate([np.expand_dims(start_state, 0), pred_traj])
        label = 'predicted distribution' if i == 0 else None
        plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'c', label=label, linewidth=4, alpha=0.1)

    # draw ground truth trajectory
    for i in range(test_num):
        # env._task._robot._robot_initial_pose = (*start_position, 0)
        # env._task._robot._robot_initial_velocity = (*start_velocity, 0)
        env.reset()

        env._task._robot.set_pose(env.physics, (*start_position, 0), np.array([1, 0, 0, 0]))
        env._task._robot.set_velocity(env.physics, (*start_velocity, 0), np.zeros(3))
        env.physics.bind(env._task._actuators).ctrl = np.zeros(2)
        # env.physics.step()
        single_traj = [start_state]
        for t in range(len(control_sequence)):
            try:
                obs = env.step(control_sequence[t])
            except:
                ipdb.set_trace()
            position = obs.observation['robot_position'][0][0:2]
            velocity = obs.observation['robot_velocity'][0][0:2]
            state = np.concatenate((position, velocity))
            single_traj.append(state)
        single_traj = np.stack(single_traj, axis=0)
        true_traj = torch.tensor(single_traj[1:, :], device=device, dtype=pred_traj_history.dtype).unsqueeze(0)
        true_traj_history = torch.cat((true_traj_history, true_traj), dim=0)
        label = 'ground truth distribution' if i == 0 else None
        plt.plot(single_traj[:, 0], single_traj[:, 1], 'r', label=label, linewidth=1, alpha=0.5)
        # plt.plot(single_traj[:, 0], single_traj[:, 1], label=label, linewidth=1, alpha=0.5)
    plt.plot(start_state[0], start_state[1], 'bo', markersize=10)
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.pause(0.05)
    plt.draw()
    plt.pause(0.05)

    # calculate the std of trajectories set
    std_true = true_traj_history.std(dim=0).mean()
    std_pred = pred_traj_history.std(dim=0).mean()

    # calculate the distance between two sets of trajectories
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
                # print(f"frechet: {i*dist.shape[0]+j}/{dist.shape[0]*dist.shape[1]}")
        dist = dist.mean()
        # print(f"!!!!!!!!!dist: {dist}!!!!!!!!!!!!")
    elif dist_type == "L2":
        # L2 distance
        pred_traj_history = pred_traj_history.reshape(pred_traj_history.shape[0], -1)
        true_traj_history = true_traj_history.reshape(true_traj_history.shape[0], -1)
        dist = torch.cdist(pred_traj_history, true_traj_history, p=2).mean()
    else:
        raise Exception(f"No such distance metric type: {dist_type}")
    flow.train()
    return dist, std_true, std_pred, prior_std

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
        # controller = RandomController([-2] * env.control_dim, [2] * env.control_dim, env.control_dim, 40)
        controller = RandomController(udim=env.control_dim, urange=1, horizon=40, lower_bound=[-2]*env.control_dim, upper_bound=[2]*env.control_dim)
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
        std_true = true_traj_history.std(axis=0).mean()
        std_pred = pred_traj_history.std(axis=0).mean()
    elif dist_type == "L2":
        # L2 distance
        pred_traj_history = pred_traj_history.reshape(pred_traj_history.shape[0], -1)
        true_traj_history = true_traj_history.reshape(true_traj_history.shape[0], -1)
        dist = torch.cdist(pred_traj_history, true_traj_history, p=2).mean()
        std_true = true_traj_history.std(dim=0).mean()
        std_pred = pred_traj_history.std(dim=0).mean()
    else:
        raise Exception(f"No such distance metric type: {dist_type}")
    flow.train()
    return dist, std_true, std_pred, prior_std

if __name__ == "__main__":
    args = parse_args()
    import time
    # environment
    env = DoubleIntegratorEnv(world_dim=args.world_dim, world_type='spheres', dt=0.05, action_noise_cov=args.action_noise*np.eye(args.world_dim))
    env_mujoco = Environment(NavigationFixObstacle())
    # random controller
    # controller = RandomController([-2] * args.world_dim, [2] * args.world_dim, args.world_dim, args.horizon)
    controller = RandomController(udim=args.control_dim, urange=10, horizon=40, lower_bound=[-10, -10], upper_bound=[10, 10])
    # flow model
    model = RealNVPModel(state_dim=args.state_dim, action_dim=args.control_dim, horizon=args.horizon, hidden_dim=args.hidden_dim, condition=args.condition_prior, flow_length=args.flow_length, initialized=True).double().to(args.device)
    utils.load_checkpoint(model, filename=args.flow_path)
    model.eval()
    dist_list = []
    for i in range(100):
        # dist, std_true, std_pred, prior_std = visualize_flow(env, model, dist_type='L2', title=args.flow_path)
        dist, std_true, std_pred, prior_std = visualize_flow_mujoco(env_mujoco, model, dist_type=args.dist_metric, test_num=args.test_num, title=args.flow_path)
        dist_list.append(dist)
        time.sleep(1)
        print(f"distance: {dist} | true traj std: {std_true} | pred traj std: {std_pred} | prior std: {prior_std.mean()}")
        # input("?")
    print(f"Average Distance over {100} tests: {np.array(dist_list).mean()}")
