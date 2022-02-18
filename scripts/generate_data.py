import argparse
import numpy as np
# from flow_mpc.environments import DoubleIntegratorEnv, QuadcopterEnv, QuadcopterDynamicEnv, VictorEnv
from flow_mpc.environments import DoubleIntegratorEnv
# from flow_mpc.controllers import FlowMPPI, RandomController
from flow_mpc.controllers import RandomController
# from flow_mpc.models import QuadcopterModel, DoubleIntegratorModel, VictorModel
from flow_mpc.models import DoubleIntegratorModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='2D_double_integrator_spheres')
    parser.add_argument('--N', type=int, default=1000, help='the number of environments')
    parser.add_argument('--planning-problems-per-env', type=int, default=100)
    parser.add_argument('--name', required=True, help='the name of the stored data file')
    parser.add_argument('--generate-trajectory', action='store_true')
    parser.add_argument('--H', type=int, default=40, help='time horizon')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--controller', type=str, default="random")
    args = parser.parse_args()

    args.env = args.env.lower()
    for (arg, value) in args._get_kwargs():
        print(f"{arg}: {value}")
    return args


def generate_planning_dataset(env, num_environments, samples_per_env, controller=None):
    data = {}
    data['starts'] = []
    data['goals'] = []
    data['sdf'] = []
    data['sdf_grad'] = []
    data['states'] = []
    if controller is not None:
        data['U'] = []
    success = False
    i = 0
    while i < num_environments:
        if (i % 100) == 0:
            print(f"{i}/{num_environments}")
        env.reset()
        starts = []
        goals = []
        control_sequences = []
        states = []     # not including the start
        sdf, sdf_grad = env.get_sdf()

        # Get multiple starts and goals for each environments
        for _ in range(samples_per_env):
            success = env.reset_start_and_goal()
            if not success:
                break

            starts.append(env.start)
            goals.append(env.goal)
            # if controller is not None:
            #     controller.update_goal(env.goal)
            #     controller.update_environment(sdf, sdf_grad)
            #     for k in range(10):
            #         #controller.controller.sigma = 3*(1 - k/10.0)
            #         _, _, control_sequence = controller.step(env.start)
            #     control_sequences.append(control_sequence)
            #     controller.controller.reset_U()
            if isinstance(controller, RandomController):
                control_sequence = controller.step(env.start)
                control_sequences.append(control_sequence)
                single_traj = []
                for t in range(len(control_sequence)):
                    env.step(control_sequence[t])
                    single_traj.append(env.state)
                single_traj = np.stack(single_traj, axis=0)
                states.append(single_traj)


        if success:
            i += 1
            data['starts'].append(np.stack(starts, axis=0))
            data['goals'].append(np.stack(goals, axis=0))
            data['sdf'].append(sdf)
            data['sdf_grad'].append(sdf_grad)
            if controller is not None:
                data['U'].append(np.stack(control_sequences, axis=0))
                data['states'].append(np.stack(states, axis=0))

    stacked_data = {}
    for key, item in data.items():
        stacked_data[key] = np.stack(item)

    return stacked_data


if __name__ == '__main__':
    args = parse_args()

    if 'victor' in args.env:
        world_dim = 3
    elif '2d' in args.env:
        world_dim = 2
    elif '3d' in args.env:
        world_dim = 3
    else:
        raise ValueError('no world dim specified')

    if 'spheres' in args.env:
        obstacles = 'spheres'
    elif 'squares' in args.env:
        obstacles = 'squares'
    elif 'narrow_passages' in args.env:
        obstacles = 'narrow_passages'
    elif 'victor' in args.env:
        pass
    else:
        raise ValueError('No obstacle type specified')

    if 'double_integrator' in args.env:
        env = DoubleIntegratorEnv(world_dim=world_dim, world_type=obstacles, dt=0.05, action_noise_cov=0.1*np.eye(world_dim))
        # generative_model = DoubleIntegratorModel(world_dim=2).to(device=args.device)
        control_dim = 2
        state_dim = 4
    else:
        raise ValueError('Env not specified appropriately')

    if args.controller is None:
        controller = None
    elif args.controller == "random":
        controller = RandomController([-2]*world_dim, [2]*world_dim, world_dim, args.H)
    else:
        raise NotImplementedError
        # controller = FlowMPPI(generative_model, args.H, control_dim, state_dim, N=1000, device='cuda:0', sigma=1.2, lambda_=1)

    env.reset()
    data = generate_planning_dataset(env, args.N, args.planning_problems_per_env, controller)

    for key, value in data.items():
        print(key, value.shape)

    np.savez(f"../data/{args.name}", **data)
    print(f"Saved to {args.name}.npz")
