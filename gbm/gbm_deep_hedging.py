import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from scipy.stats import norm
import matplotlib.pyplot as plt

import time

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
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")

    return parser.parse_args()


# Usage in your script
args = parse_args()
hidden_size = args.hidden_size
learning_rate = args.learning_rate
num_layers = args.num_layers
cycles = args.epochs

sN = 9  # 19 #9
sstep = 1  # 0.5 #1
sR = np.linspace(sstep, sstep * sN, sN)
s0 = sR[int(sN // 2)]
K = s0
smax = np.max(sR)

# trinomial model
skernel2 = np.zeros([sN, sN])
p_up = 0.2
p_down = 0.2
p_mid = 1 - p_up - p_down
skernel2 += np.eye(sN, sN, 0) * p_mid
skernel2 += np.eye(sN, sN, -1) * p_up
skernel2 += np.eye(sN, sN, 1) * p_down
skernel2[0, 0] += p_down
skernel2[sN - 1, sN - 1] += p_up

n_sim = 150_000  # this is the size of our training set for the NN
n_samples = 5  # time disrectization

fixed_cost = 0.05  # 0.05
c_treshold = 0.2

c_0 = fixed_cost / c_treshold
c_1 = 0.0  # 0.1
c_1s = fixed_cost - c_1 * c_treshold

# cost = "fixed"
cost = "2slopes"

assert cost in set(["fixed", "2slopes"])

# copied from multi3.py:
T = 5
aN = 20  # 4*20
alim = 1  # 2
astep = alim / aN  # alim/aN
bmax = T / 2 * smax * alim
bmin = -bmax  # 0 symetric (no bounds)
bstep = astep * sstep * 0.5 * 0.5
bN = int((bmax - bmin) // bstep) + 1

aR = np.linspace(0, alim, aN, endpoint=False)
bR = np.linspace(bmin, bmax, bN, endpoint=False)


# t0,si0,ai0,bi0 = 0,4,0,900*2
t0, si0, ai0, bi0 = 0, 4, 8, 900 * 2

s0 = sR[si0]
n0 = aR[ai0]

b0 = bR[bi0]

probUp = 0.2
probDown = 0.2
probMid = 1 - probUp - probDown


# here is the simplest setup of a NN
# so the option pays at maturyti, T. The NN has now many layers, which respectively output the hedging decisions
# at the individual time steps
class MinimalHedger(nn.Module):
    def __init__(self, n_samples, hidden_size=16, num_layers=3):
        super(MinimalHedger, self).__init__()
        self.contentNN = nn.ModuleList(
            [self._create_network(hidden_size, num_layers) for i in range(n_samples)]
        )

    def _create_network(self, hidden_size, num_layers):
        layers = nn.Sequential()

        # First layer (input layer)
        layers.append(nn.Linear(2, hidden_size))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(hidden_size, 1))

        return layers

    def forward(self, x):
        self.hedges = torch.zeros(x.size(), dtype=float)
        assert x.size()[1] == n_samples + 1
        for timeStep, shallowNNatDt in enumerate(self.contentNN):
            hedgeAtDt = shallowNNatDt(
                torch.cat(
                    (
                        self.hedges[:, timeStep : timeStep + 1],
                        x[:, timeStep : timeStep + 1],
                    ),
                    1,
                )
            )
            self.hedges[:, timeStep + 1] = hedgeAtDt.squeeze()
        return self.hedges


class TransformerHedger(nn.Module):
    def __init__(self, n_samples, hidden_size=32, nhead=2, num_layers=2):
        super(TransformerHedger, self).__init__()

        # Input embedding
        self.input_embedding = nn.Linear(2, hidden_size)

        # Positional encoding
        self.pos_encoder = nn.Embedding(n_samples + 1, hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output layer for hedge predictions
        self.output_layer = nn.Linear(hidden_size, 1)

        self.n_samples = n_samples
        self.hidden_size = hidden_size

    def generate_causal_mask(self, size):
        """Generate causal attention mask where each position can only attend to previous positions"""
        mask = torch.triu(torch.ones(size, size) * float("-inf"), diagonal=1)
        return mask

    def forward(self, x):
        batch_size = x.size(0)
        assert x.size(1) == self.n_samples + 1

        # Initialize hedges tensor
        self.hedges = torch.zeros(x.size(), dtype=torch.float).to(x.device)

        # Process each time step
        for t in range(self.n_samples):
            # Get sequences up to current time step
            hedge_seq = self.hedges[:, : t + 1]
            price_seq = x[:, : t + 1]

            # Concatenate features
            seq_input = torch.cat(
                [hedge_seq.unsqueeze(-1), price_seq.unsqueeze(-1)], dim=2
            )

            # Embed sequence
            embedded = self.input_embedding(seq_input)

            # Add positional encoding
            positions = torch.arange(0, t + 1, device=x.device).expand(batch_size, -1)
            embedded = embedded + self.pos_encoder(positions)

            # Apply transformer with causal masking
            mask = self.generate_causal_mask(t + 1).to(x.device)
            transformer_out = self.transformer_encoder(embedded, mask=mask)

            # Get output from final position
            final_hidden = transformer_out[:, -1]

            # Generate hedge prediction
            hedge = self.output_layer(final_hidden)

            # Update hedge value
            self.hedges[:, t + 1] = hedge.squeeze()

        return self.hedges


# Here we go, creating an artificial dataset for hedging.
# notice that we create n_sim paths of the same stock. This is a huge and unrealistic number.
# I just want to check if all works.

cycle = 0


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

        # Initialize with enough paths to fill the maximum sample size
        n_sim_adjusted = n_sim  # mmin(n_sim, max_samples)
        self.priceAtTimeZero = np.ones((n_sim_adjusted, 1), dtype=float) * b0
        self.trainX = []

        dt = 0.2
        sigma = 0.25
        mu = 0.03125

        for _ in range(n_sim):
            path = [5]
            for t in range(1, n_samples + 1):
                market_transition = path[-1] * np.exp(
                    (mu - sigma**2 / 2) * dt + sigma * np.random.normal(0, np.sqrt(dt))
                )
                path.append(market_transition)
            self.trainX.append(path)

        if self.transform:
            self.trainX = self.transform(self.trainX)

    def __len__(self):
        total_available = self.trainX.shape[0]
        paths_this_cycle = min(self.paths_per_cycle * (cycle + 1), total_available)
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

        priceAtTimeZero = torch.tensor(0)  # self.priceAtTimeZero[adjusted_idx, :]
        sample = {"trainX": trainX, "priceAtTimeZero": priceAtTimeZero}
        return sample


minimalDataset = MinimalDataset(
    n_sim=n_sim,
    n_samples=n_samples,
    total_cycles=cycles,
    paths_per_cycle=150000 // cycles,
    max_samples=(150000 // cycles) * 5,
    transform=np.log,
)
minimalDatasetTest = MinimalDataset(
    n_sim=n_sim,
    n_samples=n_samples,
    total_cycles=cycles,
    paths_per_cycle=150000 // cycles,
    max_samples=(150000 // cycles) * 5,
    transform=np.log,
)


dataLoader = torch.utils.data.DataLoader(minimalDataset, batch_size=32, shuffle=False)
dataLoaderTest = torch.utils.data.DataLoader(
    minimalDatasetTest, batch_size=1000, shuffle=False
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

minimalHedger = MinimalHedger(n_samples, hidden_size, num_layers).double()
# minimalHedger = TransformerHedger(
#     n_samples, hidden_size=hidden_size, num_layers=num_layers
# ).double()
optimizer = optim.Adam(minimalHedger.parameters(), lr=learning_rate)


def hedgingError(batch_action_ar, batch_s_ar, batchb0):
    s0 = sR[si0]
    batchb0 = batchb0.unsqueeze(-1)
    shape = batchb0.shape
    b = batchb0
    n = n0 * torch.ones(shape)
    batch_s_ar = np.exp(batch_s_ar)
    for t in range(n_samples):
        s = batch_s_ar[:, t]
        n_ = batch_action_ar[:, t + 1]
        ndiff = (n_ - n).abs()
        if cost == "fixed":
            c = fixed_cost * torch.ones(shape)
            c[ndiff < c_treshold] = 0
        if cost == "2slopes":
            c = torch.minimum(c_0 * ndiff, c_1 * ndiff + c_1s)
        b_ = b - (n_ - n) * s - c
        n = n_
        b = b_
    s = batch_s_ar[:, n_samples]
    loss = ((b + n * s - (s - K).clip(0)) ** 2).mean()
    r = 100 * (1 / s0**2) * (((b + n * s - (s - K).clip(0))) ** 2)
    r[r > 20] = 20
    r = (-1 * r) / 10 + 1
    loss = ((b + n * s - (s - K).clip(0)) ** 2).mean()
    return loss, b + n * s, s, r.mean()


# We train the Neural Network for n_epochs epochs

# we plan to compare this to mcts.
# mctsruns over 10 epochs. each epoch consists of eg 1000 episodes. those are chosen at random from the dataset of all paths
# for mcts the total number of presented paths is simply: nepochs * nepisodes. there mught be overlaps. so around 10000
start_time = time.time()
lossa = 0
ireport = 100

t_rew = []
for epoch in range(cycles):
    t = time.time()
    lossEpoch = 0
    plEpoch = 0
    epoch_values = []
    epoch_sts = []
    epoch_rewards = []
    for i, batch in enumerate(dataLoader):
        optimizer.zero_grad()
        nn_in = batch["trainX"] - np.log(s0)
        outputs = minimalHedger(nn_in.to(device))

        loss, value, sts, reward = hedgingError(
            outputs,
            batch["trainX"].to(device),
            batch["priceAtTimeZero"].unsqueeze(-1).to(device),
        )
        epoch_values.append(value[:, 0, 0].flatten().detach().numpy())
        epoch_sts.append(sts.flatten().detach().numpy())
        epoch_rewards.append(reward.flatten().detach().numpy())
        loss.backward()
        optimizer.step()
        lossEpoch += loss.item()
        lossa += loss.item()
    cycle += 1
    b2 = np.mean(
        (
            np.hstack(epoch_values)
            - np.max(
                (np.zeros(np.hstack(epoch_sts).shape), np.hstack(epoch_sts) - K), axis=0
            )
        )
        ** 2
    )
    print(
        f"Epoch {epoch}  / Loss {lossEpoch/i:.3f} / B2 {b2:.3f} / Rewards {np.mean(epoch_rewards):.3f} / Time {time.time()-t:.0f} seconds"
    )
    t_rew.append(np.mean(epoch_rewards))

first_action = outputs[0, 1].detach().numpy().item()
final_reward = np.mean(epoch_rewards)
print("first action:", first_action)
