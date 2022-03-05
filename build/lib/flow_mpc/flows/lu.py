import numpy as np
import torch

from torch import nn
from torch.nn import functional as F, init

from .linear import Linear



class LULinear(Linear):
    """A linear transform where we parameterize the LU decomposition of the weights."""

    def __init__(self, features, context_dim=None, using_cache=False, identity_init=True, eps=1e-3):
        super().__init__(features, using_cache)
        if context_dim == 0:
            context_dim = None

        self.eps = eps

        self.lower_indices = np.tril_indices(features, k=-1)
        self.upper_indices = np.triu_indices(features, k=1)
        self.diag_indices = np.diag_indices(features)

        n_triangular_entries = ((features - 1) * features) // 2

        self.lower_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))

        if context_dim is not None:
            self.net = nn.Sequential(
                nn.Linear(context_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2*n_triangular_entries)
            )
        self.unconstrained_upper_diag = nn.Parameter(torch.zeros(features))

        self._initialize(identity_init)

    def _initialize(self, identity_init):
        init.zeros_(self.bias)

        if identity_init:
            init.zeros_(self.lower_entries)
            init.zeros_(self.upper_entries)
            constant = np.log(np.exp(1 - self.eps) - 1)
            init.constant_(self.unconstrained_upper_diag, constant)
        else:
            stdv = 1.0 / np.sqrt(self.features)
            init.uniform_(self.lower_entries, -stdv, stdv)
            init.uniform_(self.upper_entries, -stdv, stdv)
            init.uniform_(self.unconstrained_upper_diag, -stdv, stdv)

    def _create_lower_upper(self, context=None):
        if context is not None and hasattr(self, 'net'):
            batch_size = context.shape[0]
            lower = self.lower_entries.new_zeros(batch_size, self.features, self.features)
            upper = self.upper_entries.new_zeros(batch_size, self.features, self.features)
            lower_entries, upper_entries = torch.chunk(self.net(context), chunks=2, dim=1)

            lower[:, self.lower_indices[0], self.lower_indices[1]] = lower_entries
            # The diagonal of L is taken to be all-ones without loss of generality.
            lower[:, self.diag_indices[0], self.diag_indices[1]] = 1.

            upper[:, self.upper_indices[0], self.upper_indices[1]] = upper_entries
            upper[:, self.diag_indices[0], self.diag_indices[1]] = self.upper_diag
        else:
            lower = self.lower_entries.new_zeros(self.features, self.features)
            upper = self.upper_entries.new_zeros(self.features, self.features)
            lower_entries, upper_entries = self.lower_entries, self.upper_entries

            lower[self.lower_indices[0], self.lower_indices[1]] = lower_entries
            # The diagonal of L is taken to be all-ones without loss of generality.
            lower[self.diag_indices[0], self.diag_indices[1]] = 1.

            upper[self.upper_indices[0], self.upper_indices[1]] = upper_entries
            upper[self.diag_indices[0], self.diag_indices[1]] = self.upper_diag

        return lower, upper

    def forward_no_cache(self, x, logpx=None, reverse=False, context=None):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D)
        where:
            D = num of features
            N = num of inputs
        """
        lower, upper = self._create_lower_upper(context)
        if reverse:
            y = F.linear(x, upper)
            y = F.linear(y, lower, self.bias)
            delta_logp = self.logabsdet() * x.new_ones(y.shape[0])
        else:
            y = x - self.bias
            y, _ = torch.triangular_solve(y.unsqueeze(-1), lower, upper=False, unitriangular=True)
            y, _ = torch.triangular_solve(y, upper, upper=True, unitriangular=False)
            y = y.squeeze(-1)
            delta_logp = -self.logabsdet() * x.new_ones(y.shape[0])

        if logpx is None:
            return y
        else:
            return y, logpx + delta_logp

    def inverse_no_cache(self, inputs):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D)
        where:
            D = num of features
            N = num of inputs
        """
        lower, upper = self._create_lower_upper()
        outputs = inputs - self.bias
        outputs, _ = torch.triangular_solve(outputs.t(), lower, upper=False, unitriangular=True)
        outputs, _ = torch.triangular_solve(outputs, upper, upper=True, unitriangular=False)
        outputs = outputs.t()

        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * inputs.new_ones(outputs.shape[0])

        return outputs, logabsdet

    def weight(self):
        """Cost:
            weight = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self._create_lower_upper()
        return lower @ upper

    def weight_inverse(self):
        """Cost:
            inverse = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self._create_lower_upper()
        identity = torch.eye(self.features, self.features)
        lower_inverse, _ = torch.trtrs(identity, lower, upper=False, unitriangular=True)
        weight_inverse, _ = torch.trtrs(lower_inverse, upper, upper=True, unitriangular=False)
        return weight_inverse

    @property
    def upper_diag(self):
        return F.softplus(self.unconstrained_upper_diag) + self.eps

    def logabsdet(self):
        """Cost:
            logabsdet = O(D)
        where:
            D = num of features
        """
        # return torch.log()
        return torch.sum(torch.log(self.upper_diag))
