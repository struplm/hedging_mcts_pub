import sys

sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class HedgerNNet(nn.Module):
    def __init__(self, game, args):
        super(HedgerNNet, self).__init__()

        self.action_size = game.getActionSize()
        self.args = args
        self.num_layers = self.args["num_layers"]
        self.num_channels = self.args["num_channels"]
        self.use_batchnorm = self.args.get("use_batchnorm", True)

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if self.use_batchnorm else None

        # Input layer
        self.layers.append(nn.Linear(25, self.num_channels))
        if self.use_batchnorm:
            self.bns.append(nn.BatchNorm1d(self.num_channels))

        # Hidden layers
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.num_channels, self.num_channels))
            if self.use_batchnorm:
                self.bns.append(nn.BatchNorm1d(self.num_channels))

        # Output layers
        self.fc_pi = nn.Linear(self.num_channels, self.action_size)
        self.fc_v = nn.Linear(self.num_channels, 1)

    def forward(self, s):
        for i, lin in enumerate(self.layers):
            s = lin(s)
            if self.use_batchnorm:
                s = self.bns[i](s)
            s = F.relu(s)
            s = F.dropout(s, p=self.args["dropout"], training=self.training)

        pi = self.fc_pi(s)
        v = self.fc_v(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
