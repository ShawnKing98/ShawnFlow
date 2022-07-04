import os.path
import pathlib
import argparse
import json
from typing import Tuple, List

import dm_control.composer.environment
import ipdb
import matplotlib.figure
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import animation, patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import copy

from dm_control.composer import Environment
from flow_mpc.environments import NavigationObstacle
from flow_mpc.environments import DoubleIntegratorEnv
from flow_mpc.controllers import RandomController
from flow_mpc.flows import *
from flow_mpc.encoders import AttentionEncoder
from flow_mpc import utils
from frechetdist import frdist

seed = 4
np.random.seed(seed)
torch.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--trial-num', type=int, default=20)
    parser.add_argument('--flow-name', type=str, default="disk_2d_free_1")
    parser.add_argument('--use-data', type=bool, default=True)
    # parser.add_argument('--data-file', type=str, default='../data/training_traj/single_disk_2d_env_2/single_disk_2d_env_2.npz')
    # parser.add_argument('--condition-prior', type=bool, default=True)
    parser.add_argument('--action-noise', type=float, default=0.1)
    parser.add_argument('--process-noise', type=float, default=0.000)
    parser.add_argument('--world-dim', type=int, default=2)
    parser.add_argument('--control-dim', type=int, default=2)
    parser.add_argument('--state-dim', type=int, default=4)
    parser.add_argument('--with-image', type=bool, default=False)
    parser.add_argument('--with-contact', type=bool, default=True)
    parser.add_argument('--double-flow', type=bool, default=False)
    # parser.add_argument('--horizon', type=int, default=20, help="The length of the future state trajectory to be considered")
    # parser.add_argument('--hidden-dim', type=int, default=256)
    # parser.add_argument('--flow-length', type=int, default=10)
    # parser.add_argument('--dist-metric', type=str, choices=['L2', 'frechet'], default='frechet',
    #                     help="the distance metric between two sets of trajectory")
    # parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.flow_path = f"../data/flow_model/{args.flow_name}/{args.flow_name}_best.pt"
    with open(f"../data/flow_model/{args.flow_name}/args.txt") as f:
        stored_args = f.read()
    stored_args = json.loads(stored_args)
    for (k, v) in stored_args.items():
        # if getattr(args, k, None) is None:
        setattr(args, k, v)
    args.dist_metric = 'frechet'
    # args.data_file = "../data/training_traj/full_disk_2d_with_contact_env_1/full_disk_2d_with_contact_env_1.npz"
    for (arg, value) in args._get_kwargs():
        print(f"{arg}: {value}")
    return args



def visualize_flow_from_data(data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                             flow: nn.Module,
                             device="cuda" if torch.cuda.is_available() else "cpu",
                             dist_type: str=None,
                             test_num: int=10,
                             horizon: int=40,
                             title=None,
                             gif_fig: matplotlib.figure.Figure=None,
                             with_contact=False,
                             with_image=True):
    """
    Visualize the predicted trajectory distribution vs. ground truth trajectory distribution of disk environment, given
    the validation dataset. Also return some stochastic metric of the two sets of trajectories.
    :param data: a dataset
    :param flow: a flow model
    :param device: running device
    :param dist_type: the type of distance metric, choice of [L2, frechet, None], None stands for no calculating at all
    :param test_num: the number of tests to be conducted in this single environment
    :param horizon: the length of the horizon to predict
    :param title: deprecated
    :param gif_fig: the matplotlib figure object to draw figure on
    :param with_contact: whether to get contact flag from the dataset
    :return dist: the average distance between two sets of trajectories
    :return std_true: the std among the ground truth trajectories
    :return std_pred: the std among the predicted trajectories
    :return prior_std: the std of the flow prior
    :return artists: (optional) a list of matplotlib artist object
    """
    flow.eval()
    idx = np.random.randint(data[0].shape[0])
    start_pose = data[0][idx, 0]
    start_tensor = torch.tensor(start_pose, device=device, dtype=torch.float).unsqueeze(0)
    action = torch.tensor(data[2][idx, 0, 0:horizon], device=device, dtype=torch.float).unsqueeze(0)
    image = data[3][idx] if with_image else None
    contact_flag = torch.tensor(data[4][idx, 0, 0:horizon], device=device, dtype=torch.float).unsqueeze(0) if with_contact else None
    artists = []
    # draw image
    if image is not None:
        image = torch.tensor(image, device=device, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        if gif_fig is not None:
            image_box = OffsetImage(1-image, cmap='gray', zoom=1.707)
            artists.append(gif_fig.axes[0].add_artist(AnnotationBbox(image_box, (0, 0), zorder=-2, pad=99)))


    # calculate the variance of the prior
    if not (isinstance(flow, DoubleImageFlowModel) or isinstance(flow, CouplingImageFlowModel)):
        prior = flow.prior
    else:
        prior = flow.dynamic_prior if with_contact else flow.contact_prior
    attn_mask = None
    if isinstance(prior, GaussianPrior):
        prior_std = np.zeros(1)
    elif isinstance(prior, ConditionalPrior):
        context = torch.cat((start_tensor, action.reshape(1, -1)), dim=1)
        if not isinstance(flow, FlowModel):
            if isinstance(flow.encoder, AttentionEncoder):
                encoder_context = torch.cat((start_tensor, action.reshape(1, -1), context.new_zeros(context.shape[0], 1)), dim=1)
                env_code, attn_mask = flow.encoder.encode(image, encoder_context)
            else:
                env_code = flow.encoder.encode(image)
            # s_u_code = flow.s_u_encoder(context)
            # context = torch.cat((s_u_code, env_code), dim=1)
            context = torch.cat((context, env_code), dim=1)
        # if contact_flag is not None:
        #     context = torch.cat((context, contact_flag), dim=1)
        context = torch.cat((context, context.new_zeros(context.shape[0], 1)), dim=1)
        # prior_mu, prior_lower = prior.mu_lower(context)
        # prior_std = (torch.det(prior_lower) ** 2).mean(0)
        # ipdb.set_trace()
        prior_mu, prior_std = torch.chunk(prior.fc(context), chunks=2, dim=1)
        prior_std = torch.sigmoid(prior_std) + 1e-7
        prior_std = prior_std.log().mean(1).mean(0)     # the log std of a single element in the prior Gaussian, rather than the std of the whole vector
    else:
        raise Exception("Unknown prior type")

    pred_traj_history = torch.zeros((0, horizon, 2), device=device)
    true_traj_history = torch.zeros((0, horizon, 2), device=device)

    # draw obstacles
    # artists.append(patches.Rectangle((-0.71, -0.71), 1.42, 1.42, linewidth=2, edgecolor='k', facecolor='none'))
    # artists.append(plt.Circle((0.04, 0.35), 0.1, color='k'))
    # artists.append(plt.Circle((-0.37, -0.33), 0.1, color='k'))
    # artists.append(plt.Circle((0.33, -0.22), 0.1, color='k'))
    # artists.append(plt.Circle((0.02, -0.4), 0.1, color='k'))
    # artists.append(plt.Circle((-0.36, 0.28), 0.1, color='k'))
    # artists.append(plt.Circle((0.40, 0.2), 0.1, color='k'))
    # artists.append(plt.Circle((-0.02, -0.03), 0.1, color='k'))
    # if gif_fig is not None:
    #     for new_obstacle in artists:
    #         gif_fig.axes[0].add_patch(new_obstacle)

    # draw predicted trajectory
    image_reconstruct = None
    for i in range(test_num):
        # ipdb.set_trace()
        with torch.no_grad():
            if isinstance(flow, FlowModel):
                pred_traj = flow(start_tensor, action)[0][:, :, 0:2]
                pred_contact_state = None
            elif isinstance(flow, CouplingImageFlowModel) or isinstance(flow, DoubleImageFlowModel):
                model_return = flow(start_tensor, action, image, reconstruct=False, contact_flag=contact_flag, output_contact=True)
                pred_traj, pred_contact, image_reconstruct = model_return[0], model_return[1], model_return[3]
                pred_traj = pred_traj[:, :, 0:2]
                pred_contact_state = pred_traj[pred_contact.bool()].double().cpu().detach().numpy()
            else:
                model_return = flow(start_tensor, action, image, reconstruct=False, contact_flag=contact_flag)
                pred_traj, image_reconstruct = model_return[0], model_return[2]
                pred_traj = pred_traj[:, :, 0:2]
                true_contact = data[4][idx, i, 0:horizon]
                pred_contact_state = pred_traj[0][true_contact].double().cpu().detach().numpy()
        pred_traj_history = torch.cat((pred_traj_history, pred_traj), dim=0)
        pred_traj = pred_traj.squeeze(0).cpu().detach().numpy()
        pred_traj = np.concatenate([np.expand_dims(start_pose[0:2], 0), pred_traj])
        label = 'predicted distribution' if i == 0 else None
        if gif_fig is not None:
            # artists += gif_fig.axes[0].plot(pred_traj[:, 0], pred_traj[:, 1], 'c', label=label, linewidth=4, alpha=0.1)
            artists += gif_fig.axes[0].plot(pred_traj[:, 0], pred_traj[:, 1], 'c', label=label, linewidth=1, alpha=0.5)
            artists.append(gif_fig.axes[0].scatter(pred_traj[:, 0], pred_traj[:, 1], 5, 'c', alpha=0.5))
            if pred_contact_state is not None:
                artists.append(gif_fig.axes[0].scatter(pred_contact_state[:, 0], pred_contact_state[:, 1], 50, 'c', 'x', alpha=0.5))
            if image_reconstruct is not None:
                image_reconstruct = image_reconstruct[0, 0, :].clip(min=0, max=1).cpu().detach().numpy()
                image_box = OffsetImage(1 - image_reconstruct, cmap='gray', zoom=1.707)
                artists.append(gif_fig.axes[0].add_artist(AnnotationBbox(image_box, (0, 0), zorder=-1, pad=99)))

    # draw ground truth trajectory
    for i in range(test_num):
        true_traj = data[1][idx, i, 0:horizon, 0:2]
        try:
            true_contact = data[4][idx, i, 0:horizon]
        except IndexError:
            true_contact = torch.zeros(horizon).bool()
        true_contact_state = true_traj[true_contact]
        single_traj = np.concatenate((np.expand_dims(start_pose[0:2], 0), true_traj), axis=0)
        label = 'ground truth distribution' if i == 0 else None
        if gif_fig is not None:
            artists += gif_fig.axes[0].plot(single_traj[:, 0], single_traj[:, 1], 'r', label=label, linewidth=1, alpha=1)
            # draw contact flags
            artists.append(gif_fig.axes[0].scatter(single_traj[:, 0], single_traj[:, 1], 5, 'r', alpha=1))
            artists.append(gif_fig.axes[0].scatter(true_contact_state[:, 0], true_contact_state[:, 1], 50, 'r', 'x', alpha=1))
    true_traj_history = torch.tensor(data[1][idx, :, 0:horizon, 0:2], device=device, dtype=torch.float)

    # draw initial position
    if gif_fig is not None:
        artists += gif_fig.axes[0].plot(start_pose[0], start_pose[1], 'bo', markersize=24)
    # if title is not None:
    #     plt.title(title)
    # plt.pause(0.5)
    # plt.draw()
    # plt.pause(0.5)

    # draw attention if needed
    if gif_fig is not None and attn_mask is not None:
        attn_mask = (10*attn_mask.mean(dim=0)).clip(max=1)
        W, H = attn_mask.shape
        for i in range(attn_mask.shape[0]):
            for j in range(attn_mask.shape[1]):
                # rectangle = plt.Rectangle((-1 + i * 2 / W, 1 - j * 2 / H), 2 / W, -2 / H, alpha=1 / (i * 32 + j + 1), color='r')
                rectangle = plt.Rectangle((-1+i*2/W, 1-j*2/H), 2/W, -2/H, alpha=attn_mask[i, j].item(), facecolor='y', edgecolor='none')
                gif_fig.axes[0].add_patch(rectangle)
                artists.append(rectangle)

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
        dist = (torch.cdist(pred_traj_history, true_traj_history, p=2).mean() / pred_traj_history.shape[1]).item()
    else:
        raise Exception(f"No such distance metric type: {dist_type}")
    flow.train()
    if gif_fig is not None:
        return dist, std_true, std_pred, prior_std, artists
    else:
        return dist, std_true, std_pred, prior_std


def visualize_flow_mujoco(env: dm_control.composer.environment.Environment,
                          flow: nn.Module,
                          controller=None,
                          horizon: int=40,
                          test_num: int=10,
                          device="cuda" if torch.cuda.is_available() else "cpu",
                          dist_type: str=None,
                          title=None,
                          gif_fig: matplotlib.figure.Figure=None):
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
        controller = RandomController(udim=control_dim, urange=10, horizon=horizon, lower_bound=[-10, -10], upper_bound=[10, 10])
    obs = env.reset()
    start_position = obs.observation['robot_position'][0][0:2]
    start_velocity = obs.observation['robot_velocity'][0][0:2]
    image = env.task.get_aerial_view(env.physics)
    start_state = np.concatenate((start_position, start_velocity))
    control_sequence = controller.step(None)
    start_tensor = torch.tensor(start_state, device=device, dtype=torch.float).unsqueeze(0)
    action = torch.tensor(control_sequence, device=device, dtype=torch.float).unsqueeze(0)
    image = torch.tensor(image, device=device, dtype=torch.float).unsqueeze(0).unsqueeze(0)

    # calculate the variance of the prior
    prior = flow.prior
    if isinstance(prior, GaussianPrior):
        prior_std = np.zeros(1)
    elif isinstance(prior, ConditionalPrior):
        context = torch.cat((start_tensor, action.reshape(1, -1)), dim=1)
        if isinstance(flow, ImageFlowModel):
            env_code = flow.encoder.encode(image)
            # s_u_code = flow.s_u_encoder(context)
            context = torch.cat((context, env_code), dim=1)
        context = torch.cat((context, context.new_zeros(1, 1)), dim=1)
        # prior_mu, prior_lower = prior.mu_lower(context)
        # prior_std = (torch.det(prior_lower) ** 2).mean(0)
        # ipdb.set_trace()
        prior_mu, prior_std = torch.chunk(prior.fc(context), chunks=2, dim=1)
        prior_std = torch.sigmoid(prior_std) + 1e-7
        prior_std = prior_std.log().mean(1).mean(0)
    else:
        raise Exception("Unknown prior type")

    pred_traj_history = torch.zeros((0, len(control_sequence), len(start_position)), device=device)
    true_traj_history = torch.zeros((0, len(control_sequence), len(start_position)), device=device)
    ldj_history = torch.zeros(0, device=device)

    # draw maze
    # f = plt.figure(0)
    # f.clf()
    # ax = f.add_subplot()
    # ax.axis('equal')
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    obstacles = []
    obstacles.append(patches.Rectangle((-0.71, -0.71), 1.42, 1.42, linewidth=2, edgecolor='k', facecolor='none'))
    for obstacle in env.task._maze.mjcf_model.find_all('geom'):
        if obstacle.name[-4:] == 'wall':
            continue
        position = obstacle.pos[0:2]
        radius = obstacle.size[0]
        obstacles.append(plt.Circle(position, radius, color='k'))
    # obstacles.append(plt.Circle((0.04, 0.35), 0.1, color='k'))
    # obstacles.append(plt.Circle((-0.37, -0.33), 0.1, color='k'))
    # obstacles.append(plt.Circle((0.33, -0.22), 0.1, color='k'))
    # obstacles.append(plt.Circle((0.02, -0.4), 0.1, color='k'))
    # obstacles.append(plt.Circle((-0.36, 0.28), 0.1, color='k'))
    # obstacles.append(plt.Circle((0.40, 0.2), 0.1, color='k'))
    # obstacles.append(plt.Circle((-0.02, -0.03), 0.1, color='k'))
    artists = copy.deepcopy(obstacles)
    # for obstacle in obstacles:
    #     ax.add_patch(obstacle)
    if gif_fig is not None:
        for new_obstacle in artists:
            gif_fig.axes[0].add_patch(new_obstacle)
    # plt.plot([-0.71, -0.71, 0.71, 0.71, -0.71], [-0.71, 0.71, 0.71, -0.71, -0.71], 'k', linewidth=2)

    # draw predicted trajectory
    for i in range(test_num):
        # ipdb.set_trace()
        with torch.no_grad():
            if isinstance(flow, FlowModel):
                pred_traj = flow(start_tensor, action)[0][:, :, 0:2]
            elif isinstance(flow, ImageFlowModel):
                pred_traj = flow(start_tensor, action, image)[0][:, :, 0:2]
        pred_traj_history = torch.cat((pred_traj_history, pred_traj), dim=0)
        pred_traj = pred_traj.squeeze(0).cpu().detach().numpy()
        pred_traj = np.concatenate([np.expand_dims(start_position, 0), pred_traj])
        label = 'predicted distribution' if i == 0 else None
        # ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'c', label=label, linewidth=4, alpha=0.1)
        if gif_fig is not None:
            artists += gif_fig.axes[0].plot(pred_traj[:, 0], pred_traj[:, 1], 'c', label=label, linewidth=1, alpha=0.5)

    # draw ground truth trajectory
    for i in range(test_num):
        # env._task._robot._robot_initial_pose = (*start_position, 0)
        # env._task._robot._robot_initial_velocity = (*start_velocity, 0)
        # env.reset()
        with env.physics.reset_context():
            env._task._robot.set_pose(env.physics, (*start_position, 0), np.array([1, 0, 0, 0]))
            env._task._robot.set_velocity(env.physics, (*start_velocity, 0), np.zeros(3))
        single_traj = [start_position]
        for t in range(len(control_sequence)):
            try:
                obs = env.step(control_sequence[t])
            except:
                ipdb.set_trace()
            position = obs.observation['robot_position'][0][0:2]
            velocity = obs.observation['robot_velocity'][0][0:2]
            state = np.concatenate((position, velocity))
            # single_traj.append(state)
            single_traj.append(position)
        single_traj = np.stack(single_traj, axis=0)
        true_traj = torch.tensor(single_traj[1:, :], device=device, dtype=pred_traj_history.dtype).unsqueeze(0)
        true_traj_history = torch.cat((true_traj_history, true_traj), dim=0)
        label = 'ground truth distribution' if i == 0 else None
        # ax.plot(single_traj[:, 0], single_traj[:, 1], 'r', label=label, linewidth=1, alpha=0.5)
        if gif_fig is not None:
            artists += gif_fig.axes[0].plot(single_traj[:, 0], single_traj[:, 1], 'r', label=label, linewidth=1, alpha=0.5)
        # plt.plot(single_traj[:, 0], single_traj[:, 1], label=label, linewidth=1, alpha=0.5)
    # ax.plot(start_state[0], start_state[1], 'bo', markersize=10)
    if gif_fig is not None:
        artists += gif_fig.axes[0].plot(start_state[0], start_state[1], 'bo', markersize=10)
    # ax.legend()
    # if title is not None:
    #     plt.title(title)
    # plt.pause(0.5)
    # plt.draw()
    # plt.pause(0.5)
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
        dist = (torch.cdist(pred_traj_history, true_traj_history, p=2).mean() / pred_traj_history.shape[1]).item()
    else:
        raise Exception(f"No such distance metric type: {dist_type}")
    flow.train()
    if gif_fig is not None:
        return dist, std_true, std_pred, prior_std, artists
    else:
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
    for i in range(10):
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
    env_mujoco = Environment(NavigationObstacle(process_noise=args.process_noise, action_noise=args.action_noise, fixed_obstacle=False), max_reset_attempts=2)
    data_dict = dict(np.load(args.data_file))
    # data_tuple = tuple(data_dict.values())
    data_list = list(zip(*data_dict.values()))
    train_val_split = int(args.train_val_ratio * len(data_list))
    train_data_tuple = tuple(zip(*data_list[0:train_val_split]))
    val_data_tuple = tuple(zip(*data_list[train_val_split:]))
    train_data_tuple = tuple(np.array(data) for data in train_data_tuple)
    val_data_tuple = tuple(np.array(data) for data in val_data_tuple)
    # env_mujoco = Environment(
    #     NavigationObstacle(process_noise=0, action_noise=1, fixed_obstacle=True),
    #     max_reset_attempts=2)
    # random controller
    controller = RandomController(udim=args.control_dim, urange=10, horizon=args.horizon, lower_bound=[-10, -10], upper_bound=[10, 10])
    # flow model
    if not args.double_flow:
        if not args.with_image:
            model = FlowModel(state_dim=args.state_dim, action_dim=args.control_dim, horizon=args.horizon, hidden_dim=args.hidden_dim,
                              condition=args.condition_prior, flow_length=args.flow_length, initialized=True, flow_type=args.flow_type).float().to(args.device)
        else:
            model = ImageFlowModel(state_dim=args.state_dim, action_dim=args.control_dim, horizon=args.horizon, image_size=(128, 128),
                                   hidden_dim=args.hidden_dim, condition=args.condition_prior, flow_length=args.flow_length,
                                   initialized=True, flow_type=args.flow_type, with_contact=False,
                                   env_dim=args.env_dim, contact_dim=args.contact_dim).float().to(args.device)
    else:
        model = DoubleImageFlowModel(state_dim=args.state_dim, action_dim=args.control_dim, horizon=args.horizon, image_size=(128, 128),
                                    hidden_dim=args.hidden_dim, condition=args.condition_prior, flow_length=args.flow_length,
                                     initialized=True, flow_type=args.flow_type, env_dim=args.env_dim).float().to(args.device)
    utils.load_checkpoint(model, filename=args.flow_path)
    model.eval()
    dist_list = []
    # test_num = 50
    ims = []
    fig = plt.figure(1)
    ax = fig.add_subplot()
    ax.set(xlim=(-1., 1.), ylim=(-1., 1.), autoscale_on=False, aspect='equal')
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.axis('equal')
    plt.close(fig)
    for i in range(args.trial_num):
        # dist, std_true, std_pred, prior_std = visualize_flow(env, model, dist_type='L2', title=args.flow_path)
        if args.use_data:
            dist, std_true, std_pred, prior_std, artists = visualize_flow_from_data(train_data_tuple, model,
                                                                                    device=args.device,
                                                                                    dist_type=args.dist_metric,
                                                                                    test_num=args.test_num,
                                                                                    horizon=args.horizon,
                                                                                    title=args.flow_path, gif_fig=fig,
                                                                                    with_contact=args.with_contact,
                                                                                    with_image=args.with_image)
        else:
            dist, std_true, std_pred, prior_std, artists = visualize_flow_mujoco(env_mujoco, model, device=args.device, controller=controller, dist_type=args.dist_metric, test_num=args.test_num, title=args.flow_path, gif_fig=fig)
        ax = plt.gca()
        ims.append(artists.copy())
        dist_list.append(dist)
        time.sleep(1)
        print(f"distance: {dist} | true traj std: {std_true} | pred traj std: {std_pred} | prior std: {prior_std.mean()}")
        # input("?")
    average_dist = np.array(dist_list).mean()
    print(f"Average Distance over {args.trial_num} tests: {average_dist}")
    ani = animation.ArtistAnimation(fig, ims, interval=3000, repeat_delay=0)
    if not os.path.exists('../data/gif'):
        os.makedirs('../data/gif')
    ani.save(f'../data/gif/{args.flow_name}.gif', writer='pillow')
    print(f"visualization result saved to data/gif/{args.flow_name}.gif.")
