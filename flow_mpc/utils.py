import torch


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth"):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)

def load_checkpoint(model, optimizer=None, filename="my_checkpoint.pth", device="cuda" if torch.cuda.is_available() else "cpu"):
    checkpoint = torch.load(filename, map_location=device)
    print(f"=> Loading checkpoint from {filename}")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
