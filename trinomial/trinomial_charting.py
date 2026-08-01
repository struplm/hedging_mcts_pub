# %%
import os
import glob
import numpy as np
import glob
import numpy as np
import matplotlib.pyplot as plt

# %%
a0_actions = []
for rr in glob.glob("ZZZ_Runs/*"):
    h_rewards_files = sorted(
        glob.glob(f"{rr}/h_rewards_*"),
        key=lambda x: int(os.path.basename(x).split("_")[2]),
    )
    latest_h_rewards_file = h_rewards_files[-1]
    data = np.load(latest_h_rewards_file, allow_pickle=True)
    a0_actions = np.hstack(
        [a0_actions, np.unique(np.array(data["actions"])[:, 0].flatten())]
    )

a0_actions = a0_actions / 20

# %%
dh_actions = []
for gg in glob.glob("./trinomial_deep_hedging*.npy"):
    dh_actions = np.hstack(
        [dh_actions, np.load(gg, allow_pickle=True).item()["results"]]
    )

# %%
fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=300)  # Same size here too


ax2.hist(
    a0_actions,
    bins=np.arange(20) / 20,
    alpha=0.7,
    edgecolor="k",
    linewidth=1.5,
    label="MCTS",
)
ax2.hist(
    dh_actions,
    bins=np.arange(20) / 20,
    alpha=0.7,
    edgecolor="k",
    linewidth=1.5,
    label="Deep Hedging",
)

ax2.set_ylabel("Frequency", fontsize=14)
ax2.set_xticks((np.array(range(20)) / 20))
ax2.set_xticklabels(["" for _ in range(20)])
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 70)

labels = np.array(range(20))
tick_positions = np.linspace(0, 1, 21)
label_positions = tick_positions[:-1] + (tick_positions[1] - tick_positions[0]) / 2
for i, label in enumerate(labels):
    ax2.text(
        label_positions[i],
        ax2.get_ylim()[0] - 0.025 * (ax2.get_ylim()[1] - ax2.get_ylim()[0]),
        str(label),
        ha="center",
        va="top",
    )

# ax2.set_xticks(range(6))
ax2.grid(axis="y", linestyle="--", alpha=0.5)
ax2.legend(loc="best", frameon=False, fontsize=12)

ax2.set_xlabel("Action taken at $t_0$", fontsize=14, labelpad=12)
fig2.tight_layout(pad=2)
plt.show()

# %%
