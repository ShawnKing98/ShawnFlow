""""Implementations of linear transforms."""

import numpy as np
import torch

from torch import nn
from torch.nn import functional as F, init

from .utils import *
__all__ = ['NaiveScale']

class Linear(nn.Module):
    """Abstract base class for linear transforms that parameterize a weight matrix."""

    def __init__(self, features, using_cache=False):
        if not is_positive_int(features):
            raise TypeError('Number of features must be a positive integer.')
        super().__init__()

        self.features = features
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x, logpx=None, reverse=False, context=None):
        return self.forward_no_cache(x, logpx, reverse, context)

    def inverse(self, inputs, context=None):
        return self.inverse_no_cache(inputs)

    def weight_and_logabsdet(self):
        # To be overridden by subclasses if it is more efficient to compute the weight matrix
        # and its logabsdet together.
        return self.weight(), self.logabsdet()

    def weight_inverse_and_logabsdet(self):
        # To be overridden by subclasses if it is more efficient to compute the weight matrix
        # inverse and weight matrix logabsdet together.
        return self.weight_inverse(), self.logabsdet()

    def forward_no_cache(self, x, logpx=None, reverse=False, context=None):
        """Applies `forward` method without using the cache."""
        raise NotImplementedError()

    def inverse_no_cache(self, inputs):
        """Applies `inverse` method without using the cache."""
        raise NotImplementedError()

    def weight(self):
        """Returns the weight matrix."""
        raise NotImplementedError()

    def weight_inverse(self):
        """Returns the inverse weight matrix."""
        raise NotImplementedError()

    def logabsdet(self):
        """Returns the log absolute determinant of the weight matrix."""
        raise NotImplementedError()


