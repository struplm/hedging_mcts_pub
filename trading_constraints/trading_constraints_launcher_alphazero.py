import logging
import os
import sys
from pathlib import Path
import coloredlogs
import yaml

import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
from Trainer import Trainer
from hedger_TV.neuralNet.trainerNeuralNet_simpleFF import (
    NNetWrapper as trainerNeuralNet,
)
from hedger_TV.hedgerGame_TV_update_corrected import HedgerPlan_TV


def main():
    # Write actual config on file
    with open(os.path.join(CONFIG["saveCheckpointsFolder"], "_config.txt"), "w") as con:
        con.write(str(CONFIG))

    scenario = HedgerPlan_TV(CONFIG)

    # Setup neural network
    nnet = trainerNeuralNet(scenario, CONFIG["nnArgs"])
    print(CONFIG)
    if CONFIG["loadModelFile"]:
        log.info('Loading model "%s/%s" ', CONFIG["loadModelFile"])
        nnet.load_checkpoint(
            CONFIG["loadModelFile"],
        )

    trainer = Trainer(scenario, nnet, CONFIG)

    if CONFIG["loadCheckpointFile"]:
        log.info('Loading data "%s/%s" ', CONFIG["loadModelFile"])
        trainer.loadTrainExamples()

    log.info("Learning.......")
    trainer.learn()


if __name__ == "__main__":
    for i in range(0, 100):
        # Load configuratio
        config_file_path = Path(os.getcwd()) / r"config/config_TV.yaml"
        if len(sys.argv) > 1:
            config_file_path = sys.argv[1]
        print(f"Reading config from: '{config_file_path}'")
        with open(config_file_path, "r") as f:
            CONFIG = yaml.safe_load(f)
        CONFIG["saveCheckpointsFolder"] = CONFIG["saveCheckpointsFolder"] + str(i)
        # Setup checkpoint folder
        checkpoints_folder = Path(CONFIG["saveCheckpointsFolder"])
        if not checkpoints_folder.exists():
            print(
                f"Checkpoint Directory '{checkpoints_folder}' does not exist. Creating it!"
            )
            checkpoints_folder.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_file_path = checkpoints_folder / "_log.txt"
        print(f"Logging to: {log_file_path}")
        logging.basicConfig(filename=log_file_path, level=logging.INFO)
        log = logging.getLogger(__name__)
        coloredlogs.install(level="INFO")

        # Run main
        main()
        logging.shutdown()

    available_actions = np.linspace(0, 2, 40, endpoint=False)
    first_actions = []
    for r in glob.glob("runs/trading_constraints/*/"):
        try:
            g = sorted(
                glob.glob(f"{r}/h_rewards_*"),
                key=lambda x: int(os.path.basename(x).split("_")[2]),
                reverse=False,
            )[-1]
            with open(g, "rb") as f:
                rrs = pickle.load(f)

            if np.mean(rrs["past"]) > np.mean(rrs["new"]):
                first_actions.append(
                    available_actions[np.unique(np.array(rrs["actionsPast"])[:, 0])]
                )
            else:
                first_actions.append(
                    available_actions[np.unique(np.array(rrs["actionsNew"])[:, 0])]
                )
        except Exception as e:
            pass

    fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=300)  # Same size here too

    ax2.hist(
        (np.hstack(first_actions * 10) / 2)[:100],
        bins=np.arange(40) / 40,
        alpha=0.7,
        edgecolor="k",
        linewidth=1.5,
        label="MCTS",
    )
    ax2.set_ylabel("Frequency", fontsize=14)
    ax2.set_xticks((np.array(range(40)) / 40))
    ax2.set_xticklabels(["" for _ in range(40)])
    ax2.set_xlim(0, 1)

    labels = np.array(range(40))
    tick_positions = np.linspace(0, 1, 41)
    label_positions = tick_positions[:-1] + (tick_positions[1] - tick_positions[0]) / 2
    for i, label in enumerate(labels):
        if i % 2 == 0:
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
