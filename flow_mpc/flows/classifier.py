import torch
from torch import nn
from torch import distributions


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(BinaryClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x):
        score = self.net(x)
        # dist = distributions.Bernoulli(score.softmax(-1)[:, :, 0])
        return score
