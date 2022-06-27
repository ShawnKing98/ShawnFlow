""" https://github.com/TheCamusean/iflow/blob/main/iflow/model/flows/coupling.py"""

import torch
import torch.nn as nn
import time
from flow_mpc import flows
# from .coupling import ResNetCouplingLayer, CouplingLayer
# from .autoregressive import MaskedAutoRegressiveLayer


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows."""
    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, context=None, reverse=False, inds=None):
        start = time.time()
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        for i in inds:
            if context is None:
                x, logpx = self.chain[i](x, logpx=logpx, reverse=reverse)
            else:
                x, logpx = self.chain[i](x, logpx=logpx, context=context, reverse=reverse)
        end = time.time()

        return x, logpx

class CouplingSequentialFlow(nn.Module):
    """Two sequential normalizing flows that have information transferred between each other during passing"""
    def __init__(self, layerList1, layerList2):
        super(CouplingSequentialFlow, self).__init__()
        self.chains = nn.ModuleList([
            nn.ModuleList(layerList1),
            nn.ModuleList(layerList2)
        ])

    def forward(self, x1, x2, logpx=None, context=None, reverse=False):
        x = [x1, x2]
        active_flow = 0
        # the possible types of layer where information is transferred
        hub_layer_class = [flows.ResNetCouplingLayer, flows.CouplingLayer, flows.MaskedAutoRegressiveLayer]
        if reverse:
            idx = [len(self.chains[0])-1, len(self.chains[1])-1]
            idx_end = [-1, -1]
            idx_step = -1
        else:
            idx = [0, 0]
            idx_end = [len(self.chains[0]), len(self.chains[1])]
            idx_step = 1
        # Keep running till both flows reach the end (might cause dead loop if the number of hub layers is different)
        while idx != idx_end:
            layer = self.chains[active_flow][idx[active_flow]]
            if type(layer) not in hub_layer_class:
                x[active_flow], logpx = layer(x[active_flow], logpx=logpx, context=context, reverse=reverse)
            else:
                other_x = x[1-active_flow]
                # Turn the contact flow output into binary variable (contact flow has to be flow 1)
                if active_flow == 1:
                    other_x = (other_x > 0).float() * 2 - 1
                if context is None:
                    full_context = other_x
                else:
                    full_context = torch.cat((other_x, context), dim=1)
                x[active_flow], logpx = layer(x[active_flow], logpx=logpx, context=full_context, reverse=reverse)
            idx[active_flow] += idx_step
            # Switch active flow when the current flow reaches an end or the next layer is a hub layer
            if idx[active_flow] == idx_end[active_flow] or type(self.chains[active_flow][idx[active_flow]]) in hub_layer_class:
                active_flow = 1-active_flow

        return x[0], x[1], logpx


if __name__ == '__main__':
    # Simple test for DoubleSequentialFlow
    flow_list_1 = []
    flow_list_2 = []
    for _ in range(3):
        flow_list_1.append(flows.MaskedAutoRegressiveLayer(horizon=5, channel=3, context_order=torch.zeros(8+5*4), intermediate_dim=128))
        flow_list_1.append(flows.ActNorm(5*3, initialized=False))
        flow_list_1.append(flows.Permutation(torch.arange(5*3).__reversed__()))

        flow_list_2.append(flows.MaskedAutoRegressiveLayer(horizon=5, channel=4, context_order=torch.zeros(8+5*3), intermediate_dim=128))
        flow_list_2.append(flows.ActNorm(5*4, initialized=False))
        flow_list_2.append(flows.Permutation(torch.arange(5*4).__reversed__()))
    flow = CouplingSequentialFlow(flow_list_1, flow_list_2)
    context = torch.randn((32, 8))
    z1 = torch.randn((32, 5*3))
    z2 = torch.randn((32, 5*4))
    logpx = torch.zeros((32,))
    res1 = flow(z1, z2, logpx=logpx, context=context, reverse=False)
    res2 = flow(z1, z2, logpx=logpx, context=context, reverse=True)
