# hedging_mcts
Repository for: Deep Hedging Under Non-Convexity: Geometry and Search-Based Diagnostics


# Reproducing Experiments:

## Creating the conda environment:

Experiments require a working python 3.11.9 conda environment. Please use the following command to create a new conda environment and install the requirements.

```bash
conda create -n <name of your environment> python=3.11.9 pip && conda activate <name of your environment> && pip install -r requirements.txt

```

## Calculating optimal action-value fnctions (Section 5.3, Section 5.4)

To produce the plots shown in the paper execute the following in a bash shell:
```bash
cd optimal_action_values
bash run
```
After execution has completed, a `fig` folder with .pdf files should have been generated.

## Learning a deterministic sequence of actions (Section 5.2)

To reproduce figure 1(b), execute the following in a bash shell:
```bash
conda activate <name of your environment>
cd fixed_action_sequence
bash run_fixed_action_sequence
```

## Learning a deterministic portfolio composition (Section 5.2)

To reproduce figure 1(c), execute the following in a bash shell:
```bash
conda activate <name of your environment>
cd fixed_portfolio_composition
bash run_fixed_portfolio_composition
```

## Learning portfolio replication in a trinomial market with non-convex costs (Section 5.3)

To reproduce figure 2(b), execute the following in a bash shell:
```bash
conda activate <name of your environment>
cd trinomial
bash run_trinomial
```


## Learning portfolio replication in a GBM market with non-convex costs (Section 5.3)

To reproduce figure 3, execute the following in a bash shell:
```bash
conda activate <name of your environment>
cd gbm
bash run_gbm
```

## Learning portfolio replication in a trinomial market with trading constraints (Section 5.4)

To reproduce figure 5(c), execute the following in a bash shell:
```bash
conda activate <name of your environment>
cd trading_constraints
python trading_constraints_launcher_alphazero.py
```
