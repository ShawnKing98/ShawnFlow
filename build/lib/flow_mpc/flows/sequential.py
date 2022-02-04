""" https://github.com/TheCamusean/iflow/blob/main/iflow/model/flows/coupling.py"""

import torch.nn as nn
import time

class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """
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