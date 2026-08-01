# dh counterexample
# in each of "n_decisions" iteration, a shallow NN tryies to guess the right actions (Q is always bimodal),
# the MDP state is here given just by iteration index
# It can be easily observed that with "n_decisions" the probability of right guess in all iterations goes to 0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from scipy.stats import norm
import matplotlib.pyplot as plt

import time
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Your script description")

    parser.add_argument(
        "--hidden_size", type=int, default=32, help="Size of hidden layers"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimization",
    )
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")

    return parser.parse_args()


args = parse_args()
hidden_size = args.hidden_size
learning_rate = args.learning_rate
num_layers = args.num_layers
cycles = args.epochs

n_decisions = 5
right_decisions = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int64)
modeval = torch.tensor([200.0, 100.0])

n_sim = 150_000  # this is the size of our training set for the NN
n_samples = 5  # time disrectization

batch_size = 32


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cycle = 0


enable_noise_input = True  # extiting shalow nets with noise
enable_unbiased_modes = True  # the modes are chosen symetricaly around 0 (unbiased), or in [0,1] interval (biased)


class JustBias(nn.Module):
    def __init__(self, in_features, out_features):
        self.biases = torch.zeros([out_features])

    def forward(self, x):
        return torch.zeros([x.shape[0], out_features]) + self.biases


def get_rewards(actions, right_decision):
    epss = modeval ** (-1)
    # modearg = 0.5*torch.tensor([right_decision, 1-right_decision])+0.25
    modearg = torch.tensor([right_decision, 1 - right_decision]) - 0.5
    # print(f"actions={actions.shape}, modearg={modearg.shape}, epss={epss.shape}")
    reward = (((actions[:, None] - modearg[None, :]) ** 2 + epss[None, :]) ** (-1)).sum(
        dim=1
    )
    # print(f"actions={actions.shape}, modearg={modearg.shape}, epss={epss.shape}, reward={reward.shape}")
    return reward


# here is the simplest setup of a NN
# so the option pays at maturyti, T. The NN has now many layers, which respectively output the hedging decisions
# at the individual time steps
class DHlikeNN(nn.Module):
    def __init__(self, n_samples, hidden_size=16, num_layers=3):
        super(DHlikeNN, self).__init__()
        self.contentNN = nn.ModuleList(
            [self._create_network(hidden_size, num_layers) for i in range(n_samples)]
        )

    def _create_network(self, hidden_size, num_layers):
        layers = nn.Sequential()

        # First layer (input layer)
        layers.append(nn.Linear(1, hidden_size))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(hidden_size, 1))

        return layers

    def forward(self, x):
        assert x.shape[1] == n_decisions
        returns = torch.zeros([x.shape[0]])
        actions = torch.zeros(x.shape)
        for t, shallowNNatDt in enumerate(self.contentNN):
            input = (
                x[:, t : t + 1]
                if enable_noise_input
                else torch.zeros(x[:, t : t + 1].shape, dtype=torch.double)
            )
            hedgeAtDt = shallowNNatDt(input)
            # print(f"t={t},x.s={x.shape},self.actions.s={self.actions.shape},shallowNNatDt[2](innerVal).s={shallowNNatDt[2](innerVal).shape}")
            # return self.actions, shallowNNatDt[2](innerVal)
            actions[:, t] = hedgeAtDt[:, 0]
            returns = returns + get_rewards(actions[:, t], right_decisions[t])
        self.returns = returns
        self.actions = actions
        return self.actions, self.returns


class MinimalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        n_sim,
        n_samples,
        total_cycles,
        paths_per_cycle=5000,
        max_samples=150000,
        transform=None,
    ):
        self.transform = transform
        self.paths_per_cycle = paths_per_cycle
        self.max_samples = max_samples
        self.total_cycles = total_cycles
        self.window_size = (
            max_samples // paths_per_cycle
        )  # Number of cycles to keep in memory

        self.data = np.random.randn(n_sim, n_samples).astype(np.double)
        # Initialize with enough paths to fill the maximum sample size
        n_sim_adjusted = n_sim  # mmin(n_sim, max_samples)
        self.priceAtTimeZero = np.ones((n_sim_adjusted, 1), dtype=float)
        if self.transform:
            self.trainX = self.transform(self.data)
        else:
            self.trainX = self.data

    def __len__(self):
        total_available = self.trainX.shape[0]
        paths_this_cycle = min(
            self.paths_per_cycle * (cycle + 1), 5 * self.paths_per_cycle
        )

        if cycle < self.window_size:
            # During initial cycles, grow up to max_samples
            return paths_this_cycle
        else:
            # After that, maintain a rolling window of size max_samples
            return min(self.max_samples, total_available)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        total_available = self.trainX.shape[0]
        paths_this_cycle = min(self.paths_per_cycle * (cycle + 1), total_available)
        if cycle < self.window_size:
            adjusted_idx = idx
        else:
            # Calculate the start of the rolling window
            window_start = (cycle - self.window_size + 1) * self.paths_per_cycle
            # Adjust for available data
            if window_start >= total_available:
                window_start = total_available - self.max_samples
            adjusted_idx = window_start + idx

        # Safety check
        if adjusted_idx >= total_available:
            adjusted_idx = adjusted_idx % total_available

        trainX = self.trainX[adjusted_idx, :]
        priceAtTimeZero = self.priceAtTimeZero[adjusted_idx, :]
        return trainX


minimalDataset = MinimalDataset(
    n_sim=n_sim,
    n_samples=n_samples,
    total_cycles=cycles,
    paths_per_cycle=150_000 // cycles,
    max_samples=(150_000 // cycles) * 5,
    transform=np.log,
)


dataLoader = torch.utils.data.DataLoader(
    minimalDataset, batch_size=batch_size, shuffle=True
)  # 32

dh_like_nn = DHlikeNN(n_decisions, hidden_size, num_layers).double().to(device)

optimizer = optim.Adam(
    dh_like_nn.parameters(), lr=learning_rate
)  # need to be switched!!! #hedger.parameters()

# torch.autograd.set_detect_anomaly(True)


start_time = time.time()
ireport = 100


for epoch in range(cycles):
    lossEpoch = 0
    epo_actions = []
    for i, batch in enumerate(dataLoader):
        optimizer.zero_grad()

        actions, returns = dh_like_nn(batch)

        loss = -returns.mean()

        with torch.no_grad():
            epo_actions.append(actions.detach().cpu().numpy())
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            lossEpoch += loss.item()

    cycle += 1


avgloss = lossEpoch / i

n_right_decisions = (-avgloss - n_decisions * modeval[1]) / (modeval[0] - modeval[1])
n_right_decisions_err = n_right_decisions - np.round(n_right_decisions)
n_right_decisions = np.round(n_right_decisions).int()
