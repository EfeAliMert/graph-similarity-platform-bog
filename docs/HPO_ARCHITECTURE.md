# Hyperparameter Optimization

HPO searches learning rate, batch size, dropout, widths, etc. Training then
fits weights with the chosen settings. One default config did not transfer
across datasets, so search is per dataset/model.

## Basic Flow

```text
dataset + model
       |
try training settings
       |
train on training pairs
       |
measure validation MSE
       |
keep the better settings
```

The optimizer minimizes validation similarity MSE. The test split is not used
to choose hyperparameters. GED error, ranking, runtime, and memory are saved as
extra information, not combined into an invented score.

The current implementation uses Optuna with the TPE sampler. Early trials
explore more freely, while later trials use previous results to choose settings
that look more promising. Poor trials may stop early.

## Budgets

The exact values are in `configs/hpo/budgets.json`.

| Budget | Trials | Confirmation seeds | Purpose |
| --- | ---: | --- | --- |
| `smoke` | 2 | 379 | Check that training works |
| `quick` | 8 | 379 | Short local search |
| `standard` | 24 | 379, 2026, 3407 | Main development search |
| `research` | 50 | 379, 2026, 3407 | Longer final search |

The strongest settings are rerun with the listed confirmation seeds. They are
ranked by mean validation MSE; standard deviation is reported separately.

## Parameters Searched

| Model | Current search |
| --- | --- |
| SimGNN | Learning rate, weight decay, dropout, batch size, GCN widths, tensor width, and histogram settings |
| GraphSim | Learning rate, batch size, patience, and pair-sampling settings |
| SEGMN | Learning rate, batch size, identity sampling, and node cap; edge cap is derived from the dataset profile |
| Graph Fusion | Learning rate, batch size, patience, and identity sampling |
| Graph2Region | Learning rate, batch size, and identity sampling |

Search space is limited to flags the trainer, checkpoint format, and adapter
actually use.

## Dataset Adaptation

Before a search, the platform measures graph count, graph size, density, labels,
and target distribution. These values change the allowed batch sizes and model
limits. For example, SEGMN uses graph-size statistics when choosing node and
edge caps because its assignment graph can require a lot of memory.

Built-in and uploaded datasets use the same profiling rules.

## Split and Reuse Rules

Graph pairs are stored with an unordered key:

```text
(min(graph_a_id, graph_b_id), max(graph_a_id, graph_b_id))
```

This prevents a pair and its reversed version from entering different
partitions. Every trial in one study uses the same validation split.

The selected configuration is saved at:

```text
configs/optimized/<dataset>/<model>.json
```

It includes the dataset fingerprint and search-space version. If the dataset,
target, split, or preprocessing changes, the old configuration is rejected.

Optuna studies and temporary checkpoints remain under:

```text
training_logs/hpo/
```

A trial checkpoint is not used as final test evidence. After HPO, the normal
workflow trains a final checkpoint with the selected settings.

## Commands

One model:

```bash
.venvs/gnn-pyg/bin/python scripts/optimize.py \
  --dataset aids700nef \
  --model simgnn \
  --budget standard \
  --seed 379
```

All five models on one dataset:

```bash
.venvs/gnn-pyg/bin/python scripts/optimize.py \
  --dataset linux \
  --model all \
  --budget quick
```

Every registered dataset except AIDS700nef:

```bash
make hpo
```

That command uses the `quick` budget, skips a row when a fingerprint-compatible
config already exists, and continues after a failed study. Configs are written
to `configs/optimized/<dataset>/<model>.json`. AIDS700nef already has tracked
configs; this matrix fills LINUX, IMDBMulti, PTC, MUTAG, PROTEINS, and ENZYMES.

HPO selects training settings. Final accuracy is measured later on held-out
test pairs and reported across the declared training seeds.

## Current Artifact State

The repository contains 35 compatible selection files: seven datasets times
five models. A selection file is evidence of the validation search, not proof
that the active checkpoint was trained from it. The checkpoint audit currently
verifies five HPO-to-weight hash bindings, all for AIDS700nef. The remaining 30
selections record `final_training: not_started`; their active checkpoints are
therefore reported as locally trained protocol checkpoints, not HPO outputs.
