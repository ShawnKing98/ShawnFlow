import argparse
import os
import json
import time
from typing import Dict, List

import ipdb
import numpy as np
from dm_control.composer import Environment
from flow_mpc.environments import NavigationObstacle, RopeManipulation
from flow_mpc.controllers import RandomController
# from flow_mpc.models import QuadcopterModel, DoubleIntegratorModel, VictorModel
from flow_mpc.models import DoubleIntegratorModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='disk_2d')
    parser.add_argument('--process-noise', type=float, default=0.000)
    parser.add_argument('--action-noise', type=float, default=0.1)
    parser.add_argument('--friction-noise', type=float, default=0.1)
    parser.add_argument('--action-range', type=float, default=10.)
    parser.add_argument('--N', type=int, default=50000, help='the number of environments')
    parser.add_argument('--samples-per-env', type=int, default=10, help='the number of sampled trajectories per environment')
    parser.add_argument('--fix-obstacle', action='store_true')
    parser.add_argument('--name', type=str, default='full_disk_2d_with_contact_env_3')
    parser.add_argument('--H', type=int, default=40, help='time horizon')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--controller', type=str, default="random")
    parser.add_argument('--remark', type=str, default='larger dataset (50,000 environments)', help="any additional information")
    args = parser.parse_args()

    args.env = args.env.lower()
    for (arg, value) in args._get_kwargs():
        print(f"{arg}: {value}")
    if not os.path.exists("../data/training_traj/" + args.name):
        os.makedirs("../data/training_traj/" + args.name)
    with open("../data/training_traj/" + args.name + "/args.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return args


def generate_disk_dataset(env, num_environments, samples_per_env, controller=None) -> Dict[str, np.ndarray]:
    data = {}
    data['starts'] = []
    data['states'] = []
    data['U'] = []
    data['views'] = []
    data['contact'] = []
    i = 0
    # control_sequence = controller.step(None)
    while i < num_environments:
        try:
            if (i % 10) == 0:
                print(f"{i}/{num_environments}")
            i += 1
            obs = env.reset()
            initial_position = obs.observation['robot_position'][0][0:2]
            initial_velocity = obs.observation['robot_velocity'][0][0:2]
            control_sequence = controller.step(None)
            starts = []
            views = env.task.get_aerial_view(env.physics)
            control_sequences = []
            states = []     # not including the start
            contact_flags = []

            # Get multiple trajectories for each environments, they have the same obstacles, start pose and control sequence
            for _ in range(samples_per_env):
                # env.reset()
                with env.physics.reset_context():
                    env._task._robot.set_pose(env.physics, (*initial_position, 0), np.array([1, 0, 0, 0]))
                    env._task._robot.set_velocity(env.physics, (*initial_velocity, 0), np.zeros(3))
                starts.append(np.concatenate((initial_position, initial_velocity)))
                control_sequences.append(control_sequence)

                single_traj = []
                single_contact_flags = []
                for t in range(len(control_sequence)):
                    obs = env.step(control_sequence[t])
                    assert obs.reward[0] is not None
                    position = obs.observation['robot_position'][0][0:2]
                    velocity = obs.observation['robot_velocity'][0][0:2]
                    single_traj.append(np.concatenate((position, velocity)))
                    single_contact_flags.append(obs.reward[0])
                single_traj = np.stack(single_traj, axis=0)
                single_contact_flags = np.stack(single_contact_flags, axis=0)
                states.append(single_traj)
                contact_flags.append(single_contact_flags)
            # ipdb.set_trace()
            data['starts'].append(np.stack(starts, axis=0))
            data['states'].append(np.stack(states, axis=0))
            data['U'].append(np.stack(control_sequences, axis=0))
            data['views'].append(views)
            data['contact'].append(contact_flags)
        except:
            i -= 1
            print("Environment broken!")

    stacked_data = {}
    for key, item in data.items():
        stacked_data[key] = np.stack(item)

    return stacked_data


def generate_rope_dataset(env, num_environments, samples_per_env, controller=None) -> Dict[str, np.ndarray]:
    data = {}
    data['starts'] = []
    data['states'] = []
    data['U'] = []
    data['views'] = []
    i = 0
    # control_sequence = controller.step(None)
    while i < num_environments:
        try:
            if (i % 1) == 0:
                print(f"{i}/{num_environments}")
            i += 1
            env._random_state.seed(i)
            env.reset()
            control_sequence = controller.step(None)
            starts = []
            views = env.task.get_aerial_view(env.physics)
            control_sequences = []
            states = []     # not including the start
            # contact_flags = []
            # Get multiple trajectories for each environments, they have the same obstacles, start pose and control sequence
            for _ in range(samples_per_env):
                env._random_state.seed(i)
                obs = env.reset()
                start_state = obs.observation['rope_pos'][0, :, 0:2].reshape(-1)
                # with env.physics.reset_context():
                #     env._task._robot.set_pose(env.physics, (*initial_position, 0), np.array([1, 0, 0, 0]))
                #     env._task._robot.set_velocity(env.physics, (*initial_velocity, 0), np.zeros(3))
                starts.append(start_state)
                control_sequences.append(control_sequence)

                single_traj = []
                single_contact_flags = []
                for t in range(len(control_sequence)):
                    obs = env.step(np.concatenate((control_sequence[t], np.zeros(1))))
                    # assert obs.reward[0] is not None
                    state = obs.observation['rope_pos'][0, :, 0:2].reshape(-1)
                    single_traj.append(state)
                    # single_contact_flags.append(obs.reward[0])
                single_traj = np.stack(single_traj, axis=0)
                # single_contact_flags = np.stack(single_contact_flags, axis=0)
                states.append(single_traj)
                # contact_flags.append(single_contact_flags)
            # ipdb.set_trace()
            data['starts'].append(np.stack(starts, axis=0))
            data['states'].append(np.stack(states, axis=0))
            data['U'].append(np.stack(control_sequences, axis=0))
            data['views'].append(views)
            # data['contact'].append(contact_flags)
        except:
            print("Environment broken!")
            i -= 1

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
    elif args.env == 'rope_2d':
        env = Environment(RopeManipulation(friction_noise=args.friction_noise))
        control_dim = 2
        state_dim = 22
    else:
        raise ValueError(f'Env {args.env} not specified appropriately')

    if args.controller is None:
        controller = None
    elif args.controller == "random":
        controller = RandomController(udim=control_dim, urange=args.action_range, horizon=args.H,
                                      lower_bound=[-args.action_range, -args.action_range],
                                      upper_bound=[args.action_range, args.action_range])
    else:
        raise NotImplementedError

    env.reset()
    if args.env == 'disk_2d':
        data = generate_disk_dataset(env, args.N, args.samples_per_env, controller)
    elif args.env == 'rope_2d':
        data = generate_rope_dataset(env, args.N, args.samples_per_env, controller)

    for key, value in data.items():
        print(key, value.shape)

    np.savez(f"../data/training_traj/{args.name}/{args.name}", **data)
    print(f"Saved to {args.name}.npz")
