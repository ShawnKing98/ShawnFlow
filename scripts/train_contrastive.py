import argparse
import json
import os
import time
import ipdb

# os.environ["CUDA_AVAILABLE_DEVICES"] = '1'

import numpy as np
import torch
from tensorboardX import SummaryWriter
import matplotlib
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
from contrastive import Aligner

np.random.seed(0)
torch.manual_seed(0)
matplotlib.use('Agg')

PROJ_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--print-epochs', type=int, default=20)
    parser.add_argument('--state-dim', type=int, default=4)
    parser.add_argument('--feature-dim', type=int, default=64)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    # parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data-file', type=str, default="full_disk_2d_with_contact_env_1", help="training data")
    # parser.add_argument('--checkpoint', type=str, default=None, help="checkpoint file and its parent folder, eg: 'test_model/test_model_20.pt' ")
    parser.add_argument('--last-epoch', type=int, default=0)
    parser.add_argument('--train-val-ratio', type=float, default=0.95)
    parser.add_argument('--name', type=str, default='disk_2d_traj_env_aligner_2', help="name of this trial")
    parser.add_argument('--remark', type=str, default='align trajectories with the environment', help="any additional information")

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
    for data in dataloader:
        # With image
        if len(data) == 4:
            start_state, traj, action, image = data
            contact_flag = None
        elif len(data) == 5:
            start_state, traj, action, contact_flag, image = data
            contact_flag = contact_flag.reshape(-1, contact_flag.shape[-1]).to(args.device)
        traj = traj.reshape(-1, *traj.shape[-2:]).to(args.device)
        image = image.reshape(-1, *image.shape[-3:]).to(args.device)
        alignment = model(image, traj, cross_align=True)
        """
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        """
        N = int(traj.shape[0] / image.shape[0])
        label_gt = torch.arange(image.shape[0]).repeat(N, 1).T.reshape(-1).to(args.device)
        loss = F.cross_entropy(alignment, label_gt)
        losses.append(loss.item())
        if backprop:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    tok = time.time()
    info = {'loss': sum(losses)/len(losses), 'time': tok-tik}
    return info


def eval_model(model, dataloader, args):
    model.eval()
    with torch.no_grad():
        info = train_model(model, dataloader, args, backprop=False)
    model.train()
    return info

if __name__ == "__main__":
    args = parse_arguments()
    writer = SummaryWriter(log_dir=os.path.join(PROJ_PATH, "scripts", "runs", args.name))

    # data
    data_dict = dict(np.load(os.path.join(PROJ_PATH, "data", "training_traj", args.data_file, args.data_file+".npz")))
    state_mean = data_dict['states'].mean((0, 1, 2))
    state_std = data_dict['states'].std((0, 1, 2))
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
    train_dataloader = DataLoader(TrajectoryImageDataset(train_data_tuple, args.horizon, with_contact=False), batch_size=args.batch_size,
                                  shuffle=True, num_workers=num_worker)
    validate_dataloader = DataLoader(TrajectoryImageDataset(val_data_tuple, args.horizon, with_contact=False), batch_size=args.batch_size,
                                     shuffle=True, num_workers=num_worker)
    del train_data_tuple
    # del val_data_tuple

    # model
    model = Aligner(feature_dim=args.feature_dim, image_size=(128, 128), state_dim=args.state_dim, horizon=args.horizon,
                state_mean=state_mean, state_std=state_std, image_mean=image_mean, image_std=image_std)
    model.to(args.device)
    model.train()
    print(f"The number of the model's trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # if args.checkpoint is not None:
    #     path = os.path.join(PROJ_PATH, "data", "flow_model", args.checkpoint)
    #     utils.load_checkpoint(model, optimizer, path, args.device)

    # train
    min_test_loss = np.inf
    best_epoch = None
    for epoch in range(args.last_epoch, args.epochs):
        train_info = train_model(model, train_dataloader, args)
        writer.add_scalar('epoch/train loss', train_info['loss'], epoch)
        print(f"epoch: {epoch} | loss: {train_info['loss']:.3g} | time: {train_info['time']:.1f} sec.")
        if epoch % args.print_epochs == 0:
            eval_info = eval_model(model, validate_dataloader, args)
            writer.add_scalar('epoch/test loss', eval_info['loss'], epoch)
            print(f"epoch: {epoch} | test loss: {eval_info['loss']:.3g} | train loss: {train_info['loss']:.3g}")
            utils.save_checkpoint(model, optimizer, os.path.join(PROJ_PATH, "data", "flow_model", args.name, f"{args.name}_{epoch}.pt"))
            if os.path.exists(os.path.join(PROJ_PATH, "data", "flow_model", args.name, f"{args.name}_{epoch-args.print_epochs}.pt")):
                os.remove(os.path.join(PROJ_PATH, "data", "flow_model", args.name, f"{args.name}_{epoch-args.print_epochs}.pt"))
            if eval_info['loss'] < min_test_loss:
                if best_epoch is not None:
                    os.remove(os.path.join(PROJ_PATH, "data", "flow_model", args.name, f"{args.name}_{best_epoch}_best.pt"))
                    # os.rename(os.path.join(PROJ_PATH, "data", "flow_model", args.name, f"{args.name}_{best_epoch}_best.pt"),
                    #             os.path.join(PROJ_PATH, "data", "flow_model", args.name, f"{args.name}_{best_epoch}.pt"))
                os.rename(os.path.join(PROJ_PATH, "data", "flow_model", args.name, f"{args.name}_{epoch}.pt"),
                            os.path.join(PROJ_PATH, "data", "flow_model", args.name, f"{args.name}_{epoch}_best.pt"))
                min_test_loss = eval_info['loss']
                best_epoch = epoch
    writer.close()
    # ipdb.set_trace()
