import torch
from torch import nn
from dm_control import mjcf


def save_checkpoint(model, optimizer, filename="my_checkpoint.pt"):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)


def load_checkpoint(model, optimizer=None, filename="my_checkpoint.pt", device="cuda" if torch.cuda.is_available() else "cpu"):
    checkpoint = torch.load(filename, map_location=device)
    print(f"=> Loading checkpoint from {filename}")
    try:
        model.load_state_dict(checkpoint["model"])
    except:
        print("model flow mean & std missing, using state mean & std")
        checkpoint['model']['flow_mean'] = checkpoint['model']['state_mean']
        checkpoint['model']['flow_std'] = checkpoint['model']['state_std']
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])


def check_body_collision(physics: mjcf.Physics, body1: str, body2: str):
    """
    Check whether the given two bodies have collision. The given body name can be either child body or parent body
    *NOTICE*: this method may be unsafe and cause hidden bug, since the way of retrieving body is to check whether
    the given name is a sub-string of the collision body
    :param physics: a MuJoCo physics engine
    :param body1: the name of body1
    :param body2: the name of body2
    :return collision: a bool variable
    """
    collision = False
    for geom1, geom2 in zip(physics.data.contact.geom1, physics.data.contact.geom2):
        bodyid1 = physics.model.geom_bodyid[geom1]
        bodyid2 = physics.model.geom_bodyid[geom2]
        bodyname1 = physics.model.id2name(bodyid1, 'body')
        bodyname2 = physics.model.id2name(bodyid2, 'body')
        if (body1 in bodyname1 and body2 in bodyname2) or (body2 in bodyname1 and body1 in bodyname2):
            collision = True
            break
    return collision

def average_grad(model: nn.Module):
    num = 0
    running_abs_grad_average = 0
    for param in model.parameters():
        new_num = param.numel()
        new_grad = param.grad.abs().mean() if param.grad is not None else 0
        running_abs_grad_average = num/(num+new_num) * running_abs_grad_average + new_num/(num+new_num) * new_grad
        num += new_num
    return running_abs_grad_average
