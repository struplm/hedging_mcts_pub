import numpy as np
from multiprocessing import Pool
import os
import argparse


def run_batch(n_runs, args):
    results = np.zeros(n_runs)
    for i in range(n_runs):
        namespace = {
            "sys": {
                "argv": [
                    "gbm_deep_hedging.py",
                    f"--hidden_size={args.hidden_size}",
                    f"--learning_rate={args.learning_rate}",
                    f"--num_layers={args.num_layers}",
                    f"--epochs={args.epochs}",
                ]
            }
        }
        with open("gbm_deep_hedging.py", "r") as file:
            exec(file.read(), namespace)
        results[i] = namespace["first_action"]
    return results


def run_parallel_analysis(args, total_runs=100, threshold=0.4):
    n_processors = os.cpu_count() - 1
    base_runs_per_proc = total_runs // n_processors
    extra_runs = total_runs % n_processors
    runs_distribution = []
    for i in range(n_processors):
        if i < extra_runs:
            runs_distribution.append(base_runs_per_proc + 1)
        else:
            runs_distribution.append(base_runs_per_proc)

    with Pool() as pool:
        partial_results = pool.starmap(
            run_batch, [(runs, args) for runs in runs_distribution]
        )

    results = np.concatenate(partial_results)
    return results


if __name__ == "__main__":

    # Fixed parameters instead of optimization
    fixed_args = argparse.Namespace(
        hidden_size=128,
        learning_rate=0.030,
        num_layers=4,
        epochs=50,
    )

    results = run_parallel_analysis(fixed_args)

    # Save the results
    np.save(
        "gbm_deep_hedging.npy",
        {
            "parameters": {
                "hidden_size": fixed_args.hidden_size,
                "learning_rate": fixed_args.learning_rate,
                "num_layers": fixed_args.num_layers,
                "epochs": fixed_args.epochs,
            },
            "results": results,
        },
    )

    print("\nAnalysis complete. Results saved to gbm_deep_hedging.npy")
