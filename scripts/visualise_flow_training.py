import pathlib
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from flow_mpc.environments import DoubleIntegratorEnv
from flow_mpc.controllers import RandomController
from flow_mpc.flows import RealNVPModel
from flow_mpc import utils


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--flow-path', type=str, default="../data/2d_integrator_no_obs_with_action_noice.pth")
    parser.add_argument('--world-dim', type=int, default=2)
    parser.add_argument('--control-dim', type=int, default=2)
    parser.add_argument('--state-dim', type=int, default=4)
    parser.add_argument('--horizon', type=int, default=40, help="The length of the future state trajectory to be considered")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return args


def visualize_flow(env, flow, controller=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    if controller is None:
        controller = RandomController([-2] * env.control_dim, [2] * env.control_dim, env.control_dim, 40)
    env.reset()
    while not env.reset_start_and_goal():
        continue
    control_sequence = controller.step(env.start)
    start_tensor = torch.tensor(env.start, device=device, dtype=torch.double).unsqueeze(0)
    action = torch.tensor(control_sequence, device=device, dtype=torch.double).unsqueeze(0)
    # traj = torch.tensor(single_traj, device=device, dtype=torch.double).unsqueeze(0)
    plt.clf()

    # draw predicted trajectory
    for i in range(100):
        pred_traj = flow(start_tensor, action)[0].squeeze(0).cpu().detach().numpy()
        pred_traj = np.concatenate([np.expand_dims(env.start, 0), pred_traj])
        label = 'prediction' if i == 0 else None
        plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'c', label=label, linewidth=4, alpha=0.3)

    # draw ground truth trajectory
    for i in range(100):
        env.state = env.start
        single_traj = [env.start]
        for t in range(len(control_sequence)):
            env.step(control_sequence[t])
            single_traj.append(env.state)
        single_traj = np.stack(single_traj, axis=0)
        label = 'ground truth' if i == 0 else None
        plt.plot(single_traj[:, 0], single_traj[:, 1], 'r', label=label, linewidth=1, alpha=0.5)
    plt.plot(env.start[0], env.start[1], 'bo', markersize=10)
    plt.legend()
    plt.pause(0.2)
    plt.show()



if __name__ == "__main__":
    args = parse_args()
    import time
    # environment
    env = DoubleIntegratorEnv(world_dim=args.world_dim, world_type='spheres', dt=0.05, action_noise_cov=0.5*np.eye(args.world_dim))
    # random controller
    controller = RandomController([-2] * args.world_dim, [2] * args.world_dim, args.world_dim, args.horizon)
    # flow model
    model = RealNVPModel(state_dim=args.state_dim, action_dim=args.control_dim, horizon=args.horizon).double().to(args.device)
    utils.load_checkpoint(model, filename=args.flow_path)
    model.eval()
    for i in range(100):
        visualize_flow(env, model)
        time.sleep(1)
    print("yes")
