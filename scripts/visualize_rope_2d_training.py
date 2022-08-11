import torch
from torch import nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation, patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import argparse
import json
import os
from typing import Tuple, List

from flow_mpc.flows import *
from flow_mpc import utils

seed = 4
np.random.seed(seed)
torch.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--trial-num', type=int, default=30)
    parser.add_argument('--flow-name', type=str, default="rope_2d_full_1")
    parser.add_argument('--use-data', type=bool, default=True)
    # parser.add_argument('--data-file', type=str, default='../data/training_traj/single_disk_2d_env_2/single_disk_2d_env_2.npz')
    # parser.add_argument('--condition-prior', type=bool, default=True)
    parser.add_argument('--action-noise', type=float, default=0.1)
    parser.add_argument('--with-image', type=bool, default=False)
    parser.add_argument('--with-contact', type=bool, default=False)
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
    args.dist_metric = 'L2'
    # args.data_file = "../data/training_traj/mul_start_disk_2d_env_3/mul_start_disk_2d_env_3.npz"
    for (arg, value) in args._get_kwargs():
        print(f"{arg}: {value}")
    return args

def visualize_rope_2d_from_data(data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                flow: nn.Module,
                                device="cuda" if torch.cuda.is_available() else "cpu",
                                dist_type: str="L2",
                                test_num: int=10,
                                horizon: int=10,
                                gif_fig: matplotlib.figure.Figure=None):
    """
        Visualize the predicted trajectory distribution vs. ground truth trajectory distribution of one environment, given
        the validation dataset. Also return some stochastic metric of the two sets of trajectories.
        :param data: a dataset
        :param flow: a flow model
        :param device: running device
        :param dist_type: the type of distance metric, choice of [L2, frechet, None], None stands for no calculating at all
        :param test_num: the number of tests to be conducted in this single environment
        :param horizon: the length of the horizon to predict
        :param gif_fig: the matplotlib figure object to draw figure on
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
    image = data[3][idx]
    # contact_flag = torch.tensor(data[4][idx, 0, 0:horizon], device=device, dtype=torch.float).unsqueeze(
    #     0) if with_contact else None
    artists = []
    # draw image
    if gif_fig is not None:
        image_box = OffsetImage(1 - image, cmap='gray', zoom=2.3)
        for i in range(horizon):
            artists.append(gif_fig.axes[i].add_artist(AnnotationBbox(image_box, (0, 0), zorder=-2, pad=0)))
    image = torch.tensor(image, device=device, dtype=torch.float).unsqueeze(0).unsqueeze(0)

    # calculate the variance of the prior
    # if not (isinstance(flow, DoubleImageFlowModel) or isinstance(flow, CouplingImageFlowModel)):
    prior = flow.prior
    # else:
    #     prior = flow.dynamic_prior if with_contact else flow.contact_prior
    if isinstance(prior, GaussianPrior):
        prior_std = np.zeros(1)
    elif isinstance(prior, ConditionalPrior):
        context = torch.cat((start_tensor, action.reshape(1, -1)), dim=1)
        if not isinstance(flow, FlowModel):
        #     if isinstance(flow.encoder, AttentionEncoder):
        #         encoder_context = torch.cat(
        #             (start_tensor, action.reshape(1, -1), context.new_zeros(context.shape[0], 1)), dim=1)
        #         env_code, attn_mask = flow.encoder.encode(image, encoder_context)
        #     else:
        #         env_code = flow.encoder.encode(image)
        #         attn_mask = None
        #     # s_u_code = flow.s_u_encoder(context)
            env_code = flow.encoder.encode(image)
            # context = torch.cat((s_u_code, env_code), dim=1)
            context = torch.cat((context, env_code), dim=1)
        # if contact_flag is not None:
        #     context = torch.cat((context, contact_flag), dim=1)
        context = torch.cat((context, context.new_zeros(context.shape[0], 1)), dim=1)
        # ipdb.set_trace()
        prior_mu, prior_std = torch.chunk(prior.fc(context), chunks=2, dim=1)
        prior_std = torch.sigmoid(prior_std) + 1e-7
        prior_std = prior_std.log().mean(1).mean(0)  # the log std of a single element in the prior Gaussian, rather than the std of the whole vector
    else:
        raise Exception("Unknown prior type")

    pred_traj_history = torch.zeros((0, horizon, start_pose.shape[0]), device=device)
    true_traj_history = torch.zeros((0, horizon, start_pose.shape[0]), device=device)

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
    # image_reconstruct = None
    for i in range(test_num):
        # ipdb.set_trace()
        with torch.no_grad():
            if isinstance(flow, FlowModel):
                pred_traj = flow(start_tensor, action)[0]
            # elif isinstance(flow, CouplingImageFlowModel) or isinstance(flow, DoubleImageFlowModel):
            #     model_return = flow(start_tensor, action, image, reconstruct=False, contact_flag=contact_flag,
            #                         output_contact=True)
            #     pred_traj, pred_contact, image_reconstruct = model_return[0], model_return[1], model_return[3]
            #     pred_traj = pred_traj[:, :, 0:2]
            #     pred_contact_state = pred_traj[pred_contact.bool()].double().cpu().detach().numpy()
            else:
                model_return = flow(start_tensor, action, image, reconstruct=False, contact_flag=None)
                pred_traj, image_reconstruct = model_return[0], model_return[2]
                pred_traj = pred_traj
                # true_contact = data[4][idx, i, 0:horizon]
                # pred_contact_state = pred_traj[0][true_contact].double().cpu().detach().numpy()
        pred_traj_history = torch.cat((pred_traj_history, pred_traj), dim=0)
        pred_traj = pred_traj.squeeze(0).cpu().detach().numpy()
        pred_traj = np.concatenate([np.expand_dims(start_pose, 0), pred_traj])
        label = 'predicted distribution' if i == 0 else None
        if gif_fig is not None:
            pred_traj = pred_traj.reshape(pred_traj.shape[0], -1, 2)
            for j in range(horizon):
                artists += gif_fig.axes[j].plot(pred_traj[j+1, :, 0], pred_traj[j+1, :, 1], 'c', label=label, linewidth=1, alpha=1)
                artists.append(gif_fig.axes[j].scatter(pred_traj[j+1, :, 0], pred_traj[j+1, :, 1], 5, 'c', alpha=1))
            # if pred_contact_state is not None:
            #     artists.append(gif_fig.axes[0].scatter(pred_contact_state[:, 0], pred_contact_state[:, 1], 50, 'c', 'x',
            #                                            alpha=0.5))
            # if image_reconstruct is not None:
            #     image_reconstruct = image_reconstruct[0, 0, :].clip(min=0, max=1).cpu().detach().numpy()
            #     image_box = OffsetImage(1 - image_reconstruct, cmap='gray', zoom=1.707)
            #     artists.append(gif_fig.axes[0].add_artist(AnnotationBbox(image_box, (0, 0), zorder=-1, pad=99)))

    # draw ground truth trajectory
    for i in range(test_num):
        true_traj = data[1][idx, i, 0:horizon]
        # true_contact = data[4][idx, i, 0:horizon]
        # true_contact_state = true_traj[true_contact]
        single_traj = np.concatenate((np.expand_dims(start_pose, 0), true_traj), axis=0)
        label = 'ground truth distribution' if i == 0 else None
        if gif_fig is not None:
            single_traj = single_traj.reshape(single_traj.shape[0], -1, 2)
            for j in range(horizon):
                artists += gif_fig.axes[j].plot(single_traj[j+1, :, 0], single_traj[j+1, :, 1], 'r', label=label, linewidth=1, alpha=1)
                artists.append(gif_fig.axes[j].scatter(single_traj[j+1, :, 0], single_traj[j+1, :, 1], 5, 'r', alpha=1))
            # # draw contact flags
            # artists.append(gif_fig.axes[0].scatter(single_traj[:, 0], single_traj[:, 1], 5, 'r', alpha=1))
            # artists.append(
            #     gif_fig.axes[0].scatter(true_contact_state[:, 0], true_contact_state[:, 1], 50, 'r', 'x', alpha=1))
    true_traj_history = torch.tensor(data[1][idx, :, 0:horizon], device=device, dtype=torch.float)

    # # draw initial position
    # if gif_fig is not None:
    #     artists += gif_fig.axes[0].plot(start_pose[0], start_pose[1], 'bo', markersize=24)
    # if title is not None:
    #     plt.title(title)
    # plt.pause(0.5)
    # plt.draw()
    # plt.pause(0.5)

    # # draw attention if needed
    # if gif_fig is not None and attn_mask is not None:
    #     attn_mask = (10 * attn_mask.mean(dim=0)).clip(max=1)
    #     W, H = attn_mask.shape
    #     for i in range(attn_mask.shape[0]):
    #         for j in range(attn_mask.shape[1]):
    #             # rectangle = plt.Rectangle((-1 + i * 2 / W, 1 - j * 2 / H), 2 / W, -2 / H, alpha=1 / (i * 32 + j + 1), color='r')
    #             rectangle = plt.Rectangle((-1 + i * 2 / W, 1 - j * 2 / H), 2 / W, -2 / H, alpha=attn_mask[i, j].item(),
    #                                       facecolor='y', edgecolor='none')
    #             gif_fig.axes[0].add_patch(rectangle)
    #             artists.append(rectangle)

    # calculate the std of trajectories set
    std_true = true_traj_history.std(dim=0).mean()
    std_pred = pred_traj_history.std(dim=0).mean()

    # calculate the distance between two sets of trajectories
    if dist_type is None:
        dist = None
    # elif dist_type == "frechet":
    #     # Frechet distance
    #     pred_traj_history = pred_traj_history.cpu().detach().numpy()
    #     true_traj_history = true_traj_history.cpu().detach().numpy()
    #     dist = np.zeros((pred_traj_history.shape[0], true_traj_history.shape[0]))
    #     for i in range(dist.shape[0]):
    #         for j in range(dist.shape[1]):
    #             dist[i, j] = frdist(pred_traj_history[i], true_traj_history[j])
    #             # print(f"frechet: {i*dist.shape[0]+j}/{dist.shape[0]*dist.shape[1]}")
    #     dist = dist.mean()
    #     # print(f"!!!!!!!!!dist: {dist}!!!!!!!!!!!!")
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

if __name__ == "__main__":
    args = parse_args()
    data_dict = dict(np.load(args.data_file))
    # data_tuple = tuple(data_dict.values())
    data_list = list(zip(*data_dict.values()))
    train_val_split = int(args.train_val_ratio * len(data_list))
    train_data_tuple = tuple(zip(*data_list[0:train_val_split]))
    val_data_tuple = tuple(zip(*data_list[train_val_split:]))
    train_data_tuple = tuple(np.array(data) for data in train_data_tuple)
    val_data_tuple = tuple(np.array(data) for data in val_data_tuple)
    if not args.with_image:
        model = FlowModel(state_dim=args.state_dim, action_dim=args.control_dim, horizon=args.horizon, hidden_dim=args.hidden_dim,
                          condition=args.condition_prior, flow_length=args.flow_length, initialized=True, flow_type=args.flow_type).float().to(args.device)
    else:
        model = ImageFlowModel(state_dim=args.state_dim, action_dim=args.control_dim, horizon=args.horizon, image_size=(128, 128),
                               hidden_dim=args.hidden_dim, condition=args.condition_prior, flow_length=args.flow_length,
                               initialized=True, flow_type=args.flow_type, with_contact=args.with_contact, env_dim=args.env_dim).float().to(args.device)
    utils.load_checkpoint(model, filename=args.flow_path)
    model.eval()
    ims = []
    fig = plt.figure(figsize=(25, 10))
    for i in range(args.horizon):
        ax = fig.add_subplot(2, 5, i+1)
        ax.set(xlim=(-1.42, 1.42), ylim=(-1.42, 1.42), autoscale_on=False, aspect='equal')
        ax.set_title(f"t=0.{i}s")
    plt.close(fig)
    dists = []
    for i in range(args.trial_num):
        dist, std_true, std_pred, prior_std, artists = visualize_rope_2d_from_data(data=val_data_tuple, flow=model,
                                                                          device=args.device,
                                                                          dist_type=args.dist_metric,
                                                                          horizon=args.horizon,
                                                                          gif_fig=fig)
        ims.append(artists)
        dists.append(dist)
        # time.sleep(1)
        print(f"distance: {dist} | true traj std: {std_true} | pred traj std: {std_pred} | prior std: {prior_std.mean()}")
    print(f"Average distance across {args.trial_num}: {sum(dists)/len(dists)}")
    ani = animation.ArtistAnimation(fig, ims, interval=8000, repeat_delay=0)
    if not os.path.exists('../data/gif'):
        os.makedirs('../data/gif')
    ani.save(f'../data/gif/{args.flow_name}.gif', writer='pillow')
    print(f"visualization result saved to data/gif/{args.flow_name}.gif.")
