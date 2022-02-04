import torch


def PointGoalFcn(pos, goal_pos):
    # state and goal can be either B x N x dx or B x dx
    goal_distance = torch.norm(pos - goal_pos, dim=-1)
    return -goal_distance


class CollisionFcn(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @classmethod
    def index_sdf(cls, sdf, indices):
        # Assume SDF is B x 1 x (sdf_dim)
        # Assume indices is B x N x dx -- dx either 2 or 3
        B, N, dx = indices.shape
        nb = torch.arange(B).view(-1, 1)
        idxs = torch.chunk(indices, chunks=dx, dim=-1)
        idxs = [ix.squeeze(-1) for ix in idxs]
        if len(idxs) == 2:
            return sdf.squeeze(1)[nb, idxs[1], idxs[0]]

        if len(idxs) == 3:
            return sdf.squeeze(1)[nb, idxs[0], idxs[1], idxs[2]]

    @staticmethod
    def forward(ctx, position, sdf, sdf_grad, check_bounds=True):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        #ctx.save_for_backward(position, sdf_grad)
        sdf_indices = (64 * (position + 2) / 4).to(dtype=torch.int64)
        collision = CollisionFcn.index_sdf(sdf, torch.clamp(sdf_indices, min=0, max=63))
        collide_val = torch.min(sdf)
        if check_bounds:
            collision = torch.where(torch.max(sdf_indices, dim=2).values > 63, collide_val * torch.ones_like(collision),collision)
            collision = torch.where(torch.min(sdf_indices, dim=2).values < 0, collide_val * torch.ones_like(collision), collision)
        return collision

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        position, sdf_grad = ctx.saved_tensors
        sdf_indices = torch.clamp((64 * (position + 2) / 4).to(dtype=torch.int64), min=0, max=63)
        position_grad = CollisionFcn.index_sdf(sdf_grad, sdf_indices)

        # TODO this needs to be dealt with -- not using gradients currently so to do later
        # out of bounds gradient
        position_grad[:, :, 0] = torch.where(position[:, :, 0] > 2, -torch.ones_like(position[:, :, 0]), position_grad[:, :, 0])
        position_grad[:, :, 0] = torch.where(position[:, :, 0] < -2, torch.ones_like(position[:, :, 0]), position_grad[:, :, 0])
        position_grad[:, :, 1] = torch.where(position[:, :, 1] > 2, -torch.ones_like(position[:, :, 0]), position_grad[:, :, 1])
        position_grad[:, :, 1] = torch.where(position[:, :, 1] < -2, torch.ones_like(position[:, :, 0]), position_grad[:, :, 1])

        velocity_grad = torch.zeros_like(position_grad)
        state_grad = position_grad * grad_output.unsqueeze(-1)
        return state_grad, None, None
