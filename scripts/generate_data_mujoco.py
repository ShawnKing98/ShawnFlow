import argparse
import os
import json
from typing import Dict, List

import ipdb
import numpy as np
from dm_control.composer import Environment
from flow_mpc.environments import NavigationObstacle
from flow_mpc.controllers import RandomController
# from flow_mpc.models import QuadcopterModel, DoubleIntegratorModel, VictorModel
from flow_mpc.models import DoubleIntegratorModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='disk_2d')
    parser.add_argument('--process-noise', type=float, default=0.000)
    parser.add_argument('--action-noise', type=float, default=0.1)
    parser.add_argument('--N', type=int, default=100, help='the number of environments')
    parser.add_argument('--samples-per-env', type=int, default=100)
    parser.add_argument('--fix-obstacle', action='store_true')
    parser.add_argument('--name', required=True, help='the name of the stored data file')
    parser.add_argument('--H', type=int, default=40, help='time horizon')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--controller', type=str, default="random")
    args = parser.parse_args()

    args.env = args.env.lower()
    for (arg, value) in args._get_kwargs():
        print(f"{arg}: {value}")
    if not os.path.exists("../data/training_traj/" + args.name):
        os.makedirs("../data/training_traj/" + args.name)
    with open("../data/training_traj/" + args.name + "/args.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return args


def generate_planning_dataset(env, num_environments, samples_per_env, controller=None) -> Dict[str, np.ndarray]:
    data = {}
    data['starts'] = []
    data['states'] = []
    data['U'] = []
    data['views'] = []
    i = 0
    control_sequence = controller.step(None)
    while i < num_environments:
        if (i % 10) == 0:
            print(f"{i}/{num_environments}")
        i += 1
        obs = env.reset()
        initial_position = obs.observation['robot_position'][0][0:2]
        initial_velocity = obs.observation['robot_velocity'][0][0:2]
        # control_sequence = controller.step(None)
        starts = []
        views = env.task.get_aerial_view(env.physics)
        control_sequences = []
        states = []     # not including the start

        # Get multiple trajectories for each environments, they have the same obstacles, start pose and control sequence
        for _ in range(samples_per_env):
            # env.reset()
            with env.physics.reset_context():
                env._task._robot.set_pose(env.physics, (*initial_position, 0), np.array([1, 0, 0, 0]))
                env._task._robot.set_velocity(env.physics, (*initial_velocity, 0), np.zeros(3))
            starts.append(np.concatenate((initial_position, initial_velocity)))
            control_sequences.append(control_sequence)

            single_traj = []
            for t in range(len(control_sequence)):
                obs = env.step(control_sequence[t])
                position = obs.observation['robot_position'][0][0:2]
                velocity = obs.observation['robot_velocity'][0][0:2]
                single_traj.append(np.concatenate((position, velocity)))
            single_traj = np.stack(single_traj, axis=0)
            states.append(single_traj)
        # ipdb.set_trace()
        data['starts'].append(np.stack(starts, axis=0))
        data['states'].append(np.stack(states, axis=0))
        data['U'].append(np.stack(control_sequences, axis=0))
        data['views'].append(views)

    stacked_data = {}
    for key, item in data.items():
        stacked_data[key] = np.stack(item)

    return stacked_data


if __name__ == '__main__':
    args = parse_args()

    if args.env == 'disk_2d':
        env = Environment(NavigationObstacle(process_noise=args.process_noise, action_noise=args.action_noise, fixed_obstacle=args.fix_obstacle), max_reset_attempts=2)
        control_dim = 2
        state_dim = 4
    else:
        raise ValueError(f'Env {args.env} not specified appropriately')

    if args.controller is None:
        controller = None
    elif args.controller == "random":
        controller = RandomController(udim=control_dim, urange=10, horizon=args.H, lower_bound=[-10, -10], upper_bound=[10, 10])
    else:
        raise NotImplementedError

    env.reset()
    data = generate_planning_dataset(env, args.N, args.samples_per_env, controller)

    for key, value in data.items():
        print(key, value.shape)

    np.savez(f"../data/training_traj/{args.name}/{args.name}", **data)
    print(f"Saved to {args.name}.npz")
