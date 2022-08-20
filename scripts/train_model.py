import argparse
import json
import os
import time
import ipdb

import numpy as np
import torch
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from torch import autograd
from torch.utils.data import DataLoader, IterableDataset
import torch.nn.functional as F
from torch import nn

from flow_mpc import flows
from flow_mpc import utils
# from flow_mpc.environments import DoubleIntegratorEnv
# from flow_mpc.environments import NavigationObstacle
# from dm_control.composer import Environment

from visualise_flow_training import visualize_flow, visualize_flow_mujoco, visualize_flow_from_data
from visualize_rope_2d_training import visualize_rope_2d_from_data
from dataset import TrajectoryDataset, TrajectoryImageDataset

np.random.seed(0)
torch.manual_seed(0)

PROJ_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--print-epochs', type=int, default=20)
    parser.add_argument('--condition-prior', type=bool, default=False)
    parser.add_argument('--with-image', type=bool, default=True)
    parser.add_argument('--state-dim', type=int, default=4)
    parser.add_argument('--control-dim', type=int, default=2)
    parser.add_argument('--env-dim', type=int, default=64)
    parser.add_argument('--double-flow', type=bool, default=False, help="whether to enable the double flow architecture")
    parser.add_argument('--with-contact', type=bool, default=True)
    parser.add_argument('--pre-rotation', type=bool, default=False)
    parser.add_argument('--contact-dim', type=int, default=1, help="the contact status dimension at one timestamp")
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--flow-length', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data-file', type=str, default="full_disk_2d_with_contact_env_1", help="training data")
    parser.add_argument('--checkpoint', type=str, default=None, help="checkpoint file and its parent folder, eg: 'test_model/test_model_20.pt' ")
    parser.add_argument('--last-epoch', type=int, default=0)
    parser.add_argument('--action-noise', type=float, default=0.1)
    parser.add_argument('--process-noise', type=float, default=0.000)
    parser.add_argument('--train-val-ratio', type=float, default=0.95)
    parser.add_argument('--flow-type', type=str, choices=['ffjord', 'nvp', 'otflow', 'autoregressive', 'msar'], default='autoregressive')
    parser.add_argument('--dist-metric', type=str, choices=['L2', 'frechet'], default='L2', help="the distance metric between two sets of trajectory")
    parser.add_argument('--name', type=str, default='disk_unconditional_autoregressive_1', help="name of this trial")
    parser.add_argument('--remark', type=str, default='ar flow with unconditional prior and latent classifier', help="any additional information")

    args = parser.parse_args()
    for (arg, value) in args._get_kwargs():
        print(f"{arg}: {value}")
    if not os.path.exists(os.path.join(PROJ_PATH, "data", "flow_model", args.name)):
        os.makedirs(os.path.join(PROJ_PATH, "data", "flow_model", args.name))
    with open(os.path.join(PROJ_PATH, "data", "flow_model", args.name, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return args

def train_model(model, dataloader, args, backprop=True):
    t0 = time.time()
    tik = t0
    losses = []
    contact_prediction_accuracy = []
    for data in dataloader:
        t1 = time.time()
        # Without image
        if isinstance(model, flows.FlowModel):
            start_state, traj, action = data
            start_state = start_state.reshape(-1, *start_state.shape[-1:]).to(args.device)
            traj = traj.reshape(-1, *traj.shape[-2:]).to(args.device)
            action = action.reshape(-1, *action.shape[-2:]).to(args.device)
            t2 = time.time()
            z, log_prob = model(start_state, action, reverse=True, traj=traj)
            t3 = time.time()
            loss = -log_prob.mean()
        # With image
        else:
            if len(data) == 4:
                start_state, traj, action, image = data
                contact_flag = None
            elif len(data) == 5:
                start_state, traj, action, contact_flag, image = data
                contact_flag = contact_flag.reshape(-1, contact_flag.shape[-1]).to(args.device)
            # N = start_state.shape[1]
            start_state = start_state.reshape(-1, *start_state.shape[-1:]).to(args.device)
            traj = traj.reshape(-1, *traj.shape[-2:]).to(args.device)
            action = action.reshape(-1, *action.shape[-2:]).to(args.device)
            # image = image.unsqueeze(1).repeat(1, N, 1, 1, 1)
            image = image.reshape(-1, *image.shape[-3:]).to(args.device)
            t2 = time.time()
            model_return = model(start_state, action, image, reconstruct=False, reverse=True, traj=traj, contact_flag=contact_flag)
            z, log_prob, image_reconstruct = model_return[0], model_return[1], model_return[2]
            loss1 = -log_prob.mean()
            # loss2 = nn.functional.mse_loss(image_reconstruct, image)
            # loss = -log_prob.mean() + 1000*nn.functional.mse_loss(image_reconstruct, image)
            loss3 = 0
            if args.contact_dim is not None:
                pred_contact_score = model_return[3]
                loss3 = F.binary_cross_entropy_with_logits(pred_contact_score, contact_flag)
                contact_prediction_accuracy.append(((pred_contact_score > 0) == contact_flag).float().mean().item())
            loss = loss1 + 5*loss3
            t3 = time.time()
        losses.append(loss.item())
        if backprop:
            # loss1.backward(retain_graph=True)
            # print("flow gradient:", utils.average_grad(model.encoder.encoder).item())
            # optimizer.zero_grad()
            # loss2.backward(retain_graph=True)
            # print("image reconstruct gradient:", utils.average_grad(model.encoder.encoder).item())
            # print("latent classification gradient:", utils.average_grad(model).item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t4 = time.time()
        tt = np.array([t0, t1, t2, t3, t4])
        dt = tt[1:]-tt[:-1]
        dt = dt/dt.sum()*100
        # print(f"sample batch: {dt[0]:.1f}% | move to cuda: {dt[1]:.1f}% | forward propagation: {dt[2]:.1f}% | back propagation: {dt[3]:.1f}%")
        t0 = time.time()
    tok = time.time()
    info = {'loss': sum(losses)/len(losses), 'time': tok-tik}
    if len(contact_prediction_accuracy) > 0:
        info['contact_prediction_accuracy'] = sum(contact_prediction_accuracy) / len(contact_prediction_accuracy)
    # print(f"train one epoch: {tok-tik} sec.")
    return info


def eval_model(model, dataloader, args):
    model.eval()
    with torch.no_grad():
        info = train_model(model, dataloader, args, backprop=False)
    model.train()
    return info

# def eval_model(model, dataloader, args):
#     model.eval()
#     loss = []
#     for data in dataloader:
#         # Without image
#         if isinstance(model, flows.FlowModel):
#             start_state, traj, action = data
#             start_state = start_state.reshape(-1, *start_state.shape[-1:]).to(args.device)
#             traj = traj.reshape(-1, *traj.shape[-2:]).to(args.device)
#             action = action.reshape(-1, *action.shape[-2:]).to(args.device)
#             with torch.no_grad():
#                 z, log_prob = model(start_state, action, reverse=True, traj=traj)
#                 loss.append(-log_prob.mean().item())
#         # With image
#         else:
#             if len(data) == 4:
#                 start_state, traj, action, image = data
#                 contact_flag = None
#             elif len(data) == 5:
#                 start_state, traj, action, contact_flag, image = data
#                 contact_flag = contact_flag.reshape(-1, contact_flag.shape[-1]).to(args.device)
#             # N = start_state.shape[1]
#             start_state = start_state.reshape(-1, *start_state.shape[-1:]).to(args.device)
#             traj = traj.reshape(-1, *traj.shape[-2:]).to(args.device)
#             action = action.reshape(-1, *action.shape[-2:]).to(args.device)
#             # image = image.unsqueeze(1).repeat(1, N, 1, 1, 1)
#             image = image.reshape(-1, *image.shape[-3:]).to(args.device)
#             with torch.no_grad():
#                 model_return = model(start_state, action, image, reconstruct=True, reverse=True, traj=traj, contact_flag=contact_flag)
#                 z, log_prob, image_reconstruct = model_return[0], model_return[1], model_return[2]
#                 loss.append((-log_prob.mean() + 1000*nn.functional.mse_loss(image_reconstruct, image)).item())
#     model.train()
#     return sum(loss)/len(loss)
if __name__ == "__main__":
    args = parse_arguments()
    writer = SummaryWriter(log_dir=os.path.join(PROJ_PATH, "scripts", "runs", args.name))

    # data
    data_dict = dict(np.load(os.path.join(PROJ_PATH, "data", "training_traj", args.data_file, args.data_file+".npz")))
    state_mean = data_dict['states'].mean((0, 1, 2))
    state_std = data_dict['states'].std((0, 1, 2))
    action_mean = data_dict['U'].mean((0, 1, 2))
    action_std = data_dict['U'].std((0, 1, 2))
    if 'views' in data_dict.keys():
        image_mean = data_dict['views'].mean()
        image_std = data_dict['views'].std()
    data_list = list(zip(*data_dict.values()))
    train_val_split = int(args.train_val_ratio * len(data_list))
    train_data_tuple = tuple(zip(*data_list[0:train_val_split]))
    val_data_tuple = tuple(zip(*data_list[train_val_split:]))
    train_data_tuple = tuple(np.array(data) for data in train_data_tuple)
    val_data_tuple = tuple(np.array(data) for data in val_data_tuple)
    del data_list
    # train_data_tuple = tuple(data_dict.values())
    # val_data_tuple = tuple(data_dict.values())
    num_worker = 0
    if not args.with_image:
        train_dataloader = DataLoader(TrajectoryDataset(train_data_tuple, args.horizon), batch_size=args.batch_size, shuffle=True, num_workers=num_worker)
        validate_dataloader = DataLoader(TrajectoryDataset(val_data_tuple, args.horizon), batch_size=args.batch_size, shuffle=True, num_workers=num_worker)
    else:
        train_dataloader = DataLoader(TrajectoryImageDataset(train_data_tuple, args.horizon, with_contact=args.with_contact), batch_size=args.batch_size,
                                      shuffle=True, num_workers=num_worker)
        validate_dataloader = DataLoader(TrajectoryImageDataset(val_data_tuple, args.horizon, with_contact=args.with_contact), batch_size=args.batch_size,
                                         shuffle=True, num_workers=num_worker)
    del train_data_tuple
    # del val_data_tuple

    # model
    if not args.double_flow:
        if not args.with_image:
            model = flows.FlowModel(state_dim=args.state_dim, action_dim=args.control_dim, horizon=args.horizon, hidden_dim=args.hidden_dim,
                                    condition=args.condition_prior, flow_length=args.flow_length, flow_type=args.flow_type,
                                    state_mean=state_mean, state_std=state_std, action_mean=action_mean, action_std=action_std).float()
        else:
            model = flows.ImageFlowModel(state_dim=args.state_dim, action_dim=args.control_dim, horizon=args.horizon, image_size=(128, 128),
                                         hidden_dim=args.hidden_dim, condition=args.condition_prior, flow_length=args.flow_length,
                                         state_mean=state_mean, state_std=state_std, action_mean=action_mean, action_std=action_std,
                                         image_mean=image_mean, image_std=image_std, flow_mean=state_mean, flow_std=state_std,
                                         flow_type=args.flow_type, with_contact=False, env_dim=args.env_dim,
                                         contact_dim=args.contact_dim, pre_rotation=args.pre_rotation).float()
    else:
        model = flows.DoubleImageFlowModel(state_dim=args.state_dim, action_dim=args.control_dim, horizon=args.horizon, image_size=(128, 128),
                                           hidden_dim=args.hidden_dim, condition=args.condition_prior, flow_length=args.flow_length,
                                           state_mean=state_mean, state_std=state_std, action_mean=action_mean, action_std=action_std,
                                           image_mean=image_mean, image_std=image_std, flow_type=args.flow_type, env_dim=args.env_dim).float()
    model.to(args.device)
    model.train()
    print(f"The number of the model's trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.checkpoint is not None:
        path = os.path.join(PROJ_PATH, "data", "flow_model", args.checkpoint)
        utils.load_checkpoint(model, optimizer, path, args.device)

    # train
    best_dist = np.inf
    best_epoch = None
    for epoch in range(args.last_epoch, args.epochs):
        train_info = train_model(model, train_dataloader, args)
        writer.add_scalar('epoch/train loss', train_info['loss'], epoch)
        contact_pred_acc = train_info['contact_prediction_accuracy'] if 'contact_prediction_accuracy' in train_info.keys() else 0
        print(f"epoch: {epoch} | loss: {train_info['loss']} | contact pred acc: {100*contact_pred_acc:.1f}% | time: {train_info['time']:.1f} sec. ")
        if epoch % args.print_epochs == 0:
            # dist, std_true, std_pred, prior_std = visualize_fn(env, model, horizon=args.horizon, dist_type=args.dist_metric, title=args.name)
            if "disk" in args.name:
                dist, std_true, std_pred, prior_std = visualize_flow_from_data(data=val_data_tuple, flow=model, device=args.device,
                                                                               dist_type=args.dist_metric, horizon=args.horizon,
                                                                               with_contact=args.with_contact, with_image=args.with_image)
            elif "rope" in args.name:
                dist, std_true, std_pred, prior_std = visualize_rope_2d_from_data(data=val_data_tuple, flow=model, device=args.device,
                                                                                  dist_type=args.dist_metric, horizon=args.horizon)
            prior_std = prior_std.mean()
            eval_info = eval_model(model, validate_dataloader, args)
            writer.add_scalar('epoch/trajectory prediction error', dist, epoch)
            writer.add_scalar('epoch/true traj std', std_true, epoch)
            writer.add_scalar('epoch/pred traj std', std_pred, epoch)
            writer.add_scalar('epoch/prior log std', prior_std, epoch)
            if 'contact_prediction_accuracy' in eval_info.keys():
                writer.add_scalar('epoch/contact prediction accuracy', eval_info['contact_prediction_accuracy'], epoch)
            writer.add_scalar('epoch/test loss', eval_info['loss'], epoch)
            print(f"epoch: {epoch} | test loss: {eval_info['loss']:.3g} | train loss: {train_info['loss']:.3g} "
                  + f"| contact pred acc: {100*contact_pred_acc:.1f}% "
                  + f"| traj prediction error: {dist:.3g} | true traj std: {std_true} | pred traj std: {std_pred} | prior log std: {prior_std}")
            utils.save_checkpoint(model, optimizer, os.path.join(PROJ_PATH, "data", "flow_model", args.name, f"{args.name}_{epoch}.pt"))
            if dist < best_dist:
                if best_epoch is not None:
                    os.rename(os.path.join(PROJ_PATH, "data", "flow_model", args.name, f"{args.name}_{best_epoch}_best.pt"),
                                os.path.join(PROJ_PATH, "data", "flow_model", args.name, f"{args.name}_{best_epoch}.pt"))
                os.rename(os.path.join(PROJ_PATH, "data", "flow_model", args.name, f"{args.name}_{epoch}.pt"),
                            os.path.join(PROJ_PATH, "data", "flow_model", args.name, f"{args.name}_{epoch}_best.pt"))
                best_dist = dist
                best_epoch = epoch
    writer.close()
    # ipdb.set_trace()
