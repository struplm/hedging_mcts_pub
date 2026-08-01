import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

n_actions = 21
actions = []
for r in glob.glob("./runs/*"):
    try:
        g = sorted(
            glob.glob(f"{r}/h_rewards_*"),
            key=lambda x: int(os.path.basename(x).split("_")[2]),
            reverse=False,
        )[-1]
        with open(g, "rb") as f:
            rrs = pickle.load(f)

        if np.mean(rrs["past"]) > np.mean(rrs["new"]):
            actions.append(rrs["actions"])
        else:
            actions.append(rrs["actions"])
    except Exception as e:
        pass

actions = np.array(actions)

fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=300)  # Same size here too


ax2.hist(
    np.unique(actions[:, :, 0], axis=1).flatten() / 20,
    bins=np.arange(20) / 20,
    alpha=0.7,
    edgecolor="k",
    linewidth=1.5,
    label="MCTS",
)
ax2.hist(
    np.load("gbm_deep_hedging.npy", allow_pickle=True).item()["results"],
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
ax2.set_ylim(0, 80)

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


ax2.grid(axis="y", linestyle="--", alpha=0.5)
ax2.legend(loc="best", frameon=False, fontsize=12)

ax2.set_xlabel("Action taken at $t_0$", fontsize=14, labelpad=12)
fig2.tight_layout(pad=2)
plt.show()
