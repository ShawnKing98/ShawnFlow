import torch
from torch.nn import DataParallel
import numpy as np
from cv2 import resize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def add_trajectory_to_axis(ax, start, goal, trajectories, sdf):
    big_sdf = resize(sdf, (256, 256))
    goal = np.clip(256 * (goal[:2] + 2) / 4, a_min=0, a_max=255).astype(np.uint64)
    start = np.clip(256 * (start[:2] + 2) / 4, a_min=0, a_max=255).astype(np.uint64)

    positions = trajectories[:, :, :2]
    positions_idx = np.clip(256 * (positions + 2) / 4, a_min=0, a_max=255).astype(np.uint64)
    ax.imshow(big_sdf[::-1])
    for i in range(len(positions_idx)):
        ax.plot(positions_idx[i, :, 0], 255 - positions_idx[i, :, 1], linewidth=0.75)
    ax.plot(goal[0], 255 - goal[1], marker='o', color="red", markersize=2)
    ax.plot(start[0], 255 - start[1], marker='o', color="blue", markersize=2)


def add_trajectory_to_axis_3d(ax, start, goal, trajectories, sdf):
    goal = np.clip(64 * (goal[:3] + 2) / 4, a_min=0, a_max=63)
    start = np.clip(64 * (start[:3] + 2) / 4, a_min=0, a_max=63)
    voxels = np.where(sdf < 0, np.ones_like(sdf), np.zeros_like(sdf)).astype(dtype=np.bool)
    ax.voxels(voxels, facecolors='k', alpha=0.25)

    positions = trajectories[:, :, :3]
    positions_idx = np.clip(63 * (positions + 2) / 4, a_min=0, a_max=63)

    for i in range(len(positions_idx)):
        ax.plot(positions_idx[i, :, 0], positions_idx[i, :, 1], positions_idx[i, :, 2], linewidth=0.75)
    ax.plot([goal[0]], [goal[1]], [goal[2]], marker='x', color="blue", markersize=3)
    ax.plot([start[0]], [start[1]], [start[2]], marker='o', color="blue", markersize=3)


def plot_trajectories(planning_network, generative_model, starts, goals, normalised_sdf, sdf, name, args):
    # Send to device
    starts = starts.to(device=args.device)
    goals = goals.to(device=args.device)
    normalised_sdf = normalised_sdf.to(device=args.device)
    sdf = sdf.to(device=args.device)

    with torch.no_grad():
        # Generate samples
        U, _, context_dict = planning_network(starts, goals, normalised_sdf, N=args.samples_per_vis)

        # Generate trajectories
        _, _, trajectory = generative_model(
            starts.unsqueeze(1).repeat(1, args.samples_per_vis, 1),
            goals.unsqueeze(1).repeat(1, args.samples_per_vis, 1),
            sdf,
            None,
            U
        )
    trajectory = trajectory.reshape(16, args.samples_per_vis, args.horizon, -1).cpu().numpy()

    d, w, h = sdf.shape[-3:]

    # Plot trajectories
    if d == 1:
        # 2d plots
        fig, axes = plt.subplots(4, 4)
        axes = axes.flatten()
        for n in range(16):
            add_trajectory_to_axis(axes[n], starts[n].cpu().numpy(), goals[n].cpu().numpy(),
                                trajectory[n], sdf[n, 0].cpu().numpy())
    else:
        fig = plt.figure()
        for n in range(16):
            ax = fig.add_subplot(4, 4, n+1, projection='3d')
            add_trajectory_to_axis_3d(ax, starts[n].cpu().numpy(), goals[n].cpu().numpy(),
                                   trajectory[n], sdf[n, 0].cpu().numpy())

    fig.savefig(f'{name}.png')
    plt.close()


def plot_sdf_samples(planning_network, normalised_sdf, name, args):
    # Send to device
    normalised_sdf = normalised_sdf.to(device=args.device)
    d, w, h = normalised_sdf.shape[-3:]
    if d > 1:
        return

    if isinstance(planning_network, DataParallel):
        encoder = planning_network.module.environment_encoder
    else:
        encoder = planning_network.environment_encoder

    # Plot Samples from flow
    with torch.no_grad():
        samples = encoder.sample(N=8)
        x = samples['environments']
        e = normalised_sdf[:, 0].cpu().numpy()
        fig, axes = plt.subplots(4, 4)
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i < 8:
                ax.imshow(e[i])
            else:
                ax.imshow(x[i - 8, 0].cpu().numpy())

    fig.savefig(f'{name}.png')
    plt.close()


def plot_sdf_reconstructions(planning_network, normalised_sdf, name, args):
    # Send to device
    normalised_sdf = normalised_sdf.to(device=args.device)
    d, w, h = normalised_sdf.shape[-3:]
    if d > 1:
        return
    # Plot possible reconstructions from z_env that is used for planning

    if isinstance(planning_network, DataParallel):
        encoder = planning_network.module.environment_encoder
    else:
        encoder = planning_network.environment_encoder

    with torch.no_grad():
        e = normalised_sdf[:, 0].cpu().numpy()
        out = encoder.encode(normalised_sdf)
        z_env = out['z_environment']
        reconstructed_envs = encoder.reconstruct(z_env, N=4)
        reconstructed_envs = reconstructed_envs['environments']
        fig, axes = plt.subplots(4, 5)
        for i in range(4):
            axes[i][0].imshow(e[i])

            for j in range(4):
                axes[i][j + 1].imshow(reconstructed_envs[j, i, 0].cpu().numpy())

    fig.savefig(f'{name}.png')
    plt.close()


def plot_sdf(sdf, name):

    sdf = sdf.detach().cpu().numpy()
    d, w, h = sdf.shape[-3:]
    if d > 1:
        return

    fig, axes = plt.subplots(4, 4)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(sdf[i, 0])

    fig.savefig(f'{name}.png')
    plt.close()
