from flow_mpc.flows import ImageFlowModel
from flow_mpc import utils
from flow_mpc.controllers import mppi
from flow_mpc.environments import NavigationObstacle
from dm_control import composer
import torch
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from matplotlib import animation, patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import time


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--flow-name', type=str, default="autoregressive_full_13")
    parser.add_argument('--sample-num', type=int, default=100)
    parser.add_argument('--use-data', type=bool, default=True)
    # parser.add_argument('--data-file', type=str, default='../data/training_traj/single_disk_2d_env_2/single_disk_2d_env_2.npz')
    # parser.add_argument('--condition-prior', type=bool, default=True)
    # parser.add_argument('--action-noise', type=float, default=0.1)
    # parser.add_argument('--process-noise', type=float, default=0.000)
    # parser.add_argument('--world-dim', type=int, default=2)
    parser.add_argument('--control-dim', type=int, default=2)
    parser.add_argument('--state-dim', type=int, default=4)
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
    args.dist_metric = 'frechet'
    # args.data_file = "../data/training_traj/mul_start_disk_2d_env_3/mul_start_disk_2d_env_3.npz"
    for (arg, value) in args._get_kwargs():
        print(f"{arg}: {value}")
    return args


class FlowCostFunction:
    """
    A cost function that uses flow model to predict the future and encourages the robot to move towards the goal
    """
    def __init__(self, flow: torch.nn.Module, goal=None, Q=(10, 10, 0., 0.), R=(0., 0.), env_image=None, test_num: int = 10, device="cuda"):
        self._flow = flow
        self._goal = goal
        self._env_image = env_image
        self._test_num = test_num
        self._device = device
        self._Q = torch.tensor(Q, device=device, dtype=torch.float)
        self._R = torch.tensor(R, device=device, dtype=torch.float)

    def __call__(self, x, action):
        x = torch.tensor(x, device=self._device, dtype=torch.float)
        if type(action) is not torch.Tensor:
            action = torch.tensor(action, device=self._device, dtype=torch.float)
        traj_average = x.new_zeros((action.shape[0], action.shape[1], x.shape[-1]))

        with torch.no_grad():
            for _ in range(self._test_num):
                traj, prob = self._flow(x.repeat(action.shape[0], 1), action, self._env_image)
                traj_average += traj
        traj_average /= self._test_num
        state_goal = torch.cat((self._goal, torch.zeros_like(self._goal)))
        cost = (((traj_average - state_goal) ** 2) @ self._Q).sum(dim=-1)
        cost += ((traj_average[:, -1] - state_goal) ** 2) @ (5*self._Q)
        cost += (action @ self._R).sum(dim=-1)
        # final_pos = final_state[:, 0:2]
        # final_vel = final_state[:, 2:4]
        # cost = torch.linalg.norm(final_pos-self._goal, dim=1) + torch.linalg.norm(final_vel, dim=1)
        # print(f"cost: {cost}")
        return cost

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, value):
        self._goal = torch.tensor(value, device=self._device, dtype=torch.float)

    @property
    def env_image(self):
        return self._env_image.cpu().detach().numpy()

    @env_image.setter
    def env_image(self, value):
        self._env_image = torch.tensor(value, device=self._device, dtype=torch.float)
        while self._env_image.dim() < 4:
            self._env_image.unsqueeze_(0)


if __name__ == "__main__":
    args = parse_args()
    # seed = 2    # typical failure case
    seed = 0
    flow = ImageFlowModel(state_dim=args.state_dim, action_dim=args.control_dim, channel_num=args.state_dim, horizon=args.horizon, image_size=(128, 128),
                          hidden_dim=args.hidden_dim, condition=True, flow_length=args.flow_length, initialized=True,
                          flow_type=args.flow_type, with_contact=False, env_dim=args.env_dim).float().to(args.device)
    utils.load_checkpoint(model=flow, filename=args.flow_path, device=args.device)
    cost_function = FlowCostFunction(flow)
    task = NavigationObstacle(process_noise=args.process_noise, action_noise=args.action_noise, fixed_obstacle=False, cost_function=cost_function)
    # seed = np.random.RandomState(22)

    env = composer.Environment(task, random_state=seed, max_reset_attempts=2)
    control_limit = torch.tensor((10., 10.), device="cuda")
    planner = mppi.MPPI(cost=cost_function, dx=args.state_dim, du=args.control_dim, horizon=args.horizon, num_samples=100, lambda_=0.1, sigma=10,
                        control_constraints=(-control_limit, control_limit), device="cuda")

    fig = plt.figure(0)
    ax = fig.add_subplot()
    ax.set(xlim=(-1., 1.), ylim=(-1., 1.), autoscale_on=False, aspect='equal')
    time.sleep(0.5)
    plt.close(fig)
    artists = []
    obs = env.reset()
    time.sleep(0.5)
    image = cost_function.env_image[0, 0]
    goal = cost_function.goal.cpu().detach().numpy()
    print("Start planning...")
    for i in range(50):
        print(f"planning step {i}")
        robot_pos = obs.observation['robot_position'][0, 0:2]
        robot_vel = obs.observation['robot_velocity'][0, 0:2]
        robot_state = np.concatenate((robot_pos, robot_vel), axis=0)
        action_seq = planner.step(robot_state)
        action = action_seq[0].cpu().detach().numpy()
        print(f"action: {action}")

        # draw environment
        image_box = OffsetImage(1-image, cmap='gray', zoom=1.707)
        artist = [ax.add_artist(AnnotationBbox(image_box, (0, 0), zorder=-1, pad=99))]

        # draw top K traj option
        top_k_actions = planner.best_K_U
        # print(f"top k action: {top_k_actions}")
        s0_tensor = torch.tensor(robot_state, device=args.device, dtype=torch.float)
        traj_average = 0
        for _ in range(10):
            traj, prob = cost_function._flow(s0_tensor.repeat(top_k_actions.shape[0], 1), top_k_actions, cost_function._env_image)
            traj_average = traj_average + traj
        traj_average = (traj_average / 10).cpu().detach().numpy()
        # print(traj.shape)
        for single_traj in traj_average:
            # print(single_traj)
            artist += ax.plot(single_traj[:, 0], single_traj[:, 1], 'r', label=None, linewidth=1, alpha=0.5)

        #draw robot and goal
        artist += ax.plot(robot_pos[0], robot_pos[1], 'bo', markersize=22, alpha=1)
        artist += ax.plot(goal[0], goal[1], 'y*', markersize=8)

        artists.append(artist)

        obs = env.step(action)
    env.close()

    ani = animation.ArtistAnimation(fig, artists, interval=1000, repeat_delay=3000)
    ani.save(f'../data/gif/mppi_{args.flow_name}.gif', writer='pillow')
    print(f"visualization result saved to data/gif/mppi_{args.flow_name}.gif.")
