# Research Results

These are local-checkpoint results. They are not reproductions of the five papers' published accuracy tables.

Generated: `2026-08-30T21:11:33.527247+00:00`.
Matrix complete: `True`.
Mode: `evaluate-existing`.
Held-out pairs: `50`.

Exact A* GED and approximate solver upper bounds stay in separate tables. A standard deviation is not estimable from one evaluation seed; the table therefore reports pair-bootstrap 95% confidence intervals for GED MAE. Paper-level MCS and graph-classification experiments remain out of scope.

## Exact A* GED

| Dataset | Model | Eval seeds | Checkpoint seeds | GED MAE | Pair-bootstrap 95% CI | MSE x1e3 | Spearman | NDCG@k | Split verified |
|---|---|---|---|---:|---|---:|---:|---:|---|
| aids700nef | Graph Fusion | 379 | 379 | 1.210825 | [0.886, 1.500] | 4.794403 | 0.81925 | 0.932977 | True |
| aids700nef | Graph2Region | 379 | 379 | 2.42162 | [1.827, 3.102] | 21.924146 | 0.510979 | 0.8152 | True |
| aids700nef | Multi-Scale Convolutional Set Matching | 379 | 379 | 2.413357 | [1.928, 3.070] | 18.419311 | 0.586345 | 0.896534 | True |
| aids700nef | SEGMN | 379 | 379 | 1.56415 | [1.076, 2.203] | 4.905557 | 0.750471 | 0.921881 | True |
| aids700nef | SimGNN | 379 | 379 | 1.830641 | [1.431, 2.246] | 13.132869 | 0.704635 | 0.904134 | True |
| linux | Graph Fusion | 379 | 3407 | 0.805275 | [0.578, 1.061] | 4.452213 | 0.935768 | 0.996252 | True |
| linux | Graph2Region | 379 | 379 | 1.360405 | [1.058, 1.719] | 10.609648 | 0.821375 | 0.952716 | True |
| linux | Multi-Scale Convolutional Set Matching | 379 | 3407 | 1.084162 | [0.876, 1.299] | 11.012762 | 0.908221 | 0.935328 | True |
| linux | SEGMN | 379 | 379 | 0.573866 | [0.452, 0.714] | 2.251903 | 0.965879 | 0.989106 | True |
| linux | SimGNN | 379 | 3407 | 1.006579 | [0.784, 1.242] | 11.412284 | 0.915048 | 0.775788 | True |

## Approximate GED benchmark

| Dataset | Model | Eval seeds | Checkpoint seeds | GED MAE | Pair-bootstrap 95% CI | MSE x1e3 | Spearman | NDCG@k | Split verified |
|---|---|---|---|---:|---|---:|---:|---:|---|
| imdbmulti | Graph Fusion | 379 | 379 | 70.390645 | [26.140, 131.904] | 21.999862 | 0.824886 | 0.914508 | True |
| imdbmulti | Graph2Region | 379 | 379 | 80.12518 | [34.620, 147.206] | 14.975775 | 0.912899 | 0.924731 | True |
| imdbmulti | Multi-Scale Convolutional Set Matching | 379 | 379 | 68.412474 | [28.807, 133.103] | 13.102723 | 0.881937 | 0.965956 | True |
| imdbmulti | SEGMN | 379 | 379 | 54.471122 | [24.221, 102.805] | 8.803353 | 0.898583 | 0.965299 | True |
| imdbmulti | SimGNN | 379 | 379 | 81.0877 | [42.090, 137.753] | 87.589611 | 0.935767 | 0.917872 | True |
| ptc | Graph Fusion | 379 | 379 | 17.114154 | [7.565, 32.350] | 8.960377 | 0.935366 | 0.936387 | True |
| ptc | Graph2Region | 379 | 379 | 21.334525 | [15.989, 27.670] | 31.182235 | 0.560144 | 0.614522 | True |
| ptc | Multi-Scale Convolutional Set Matching | 379 | 379 | 8.013453 | [6.249, 10.352] | 11.186433 | 0.950444 | 0.878565 | True |
| ptc | SEGMN | 379 | 379 | 14.838236 | [9.365, 21.618] | 19.463081 | 0.801124 | 0.889526 | True |
| ptc | SimGNN | 379 | 379 | 42.700638 | [16.434, 82.006] | 10.589863 | 0.932677 | 0.959177 | True |

## Retrieval

Structural prefilter first. GNN reranking scores only the surviving budget.

### Tie-aware structural prefilter

| Dataset | Relevant pairs | Budget | Recall@k | GED regret | Reduction % |
|---|---:|---:|---:|---:|---:|
| aids700nef | 10 | 1 | 0.0 | 6.0 | 99.999 |
| aids700nef | 10 | 4 | 0.0 | 2.0 | 99.995 |
| aids700nef | 10 | 8 | 0.0 | 2.0 | 99.99 |
| aids700nef | 10 | 16 | 0.0 | 2.0 | 99.98 |
| aids700nef | 10 | 32 | 0.0 | 2.0 | 99.959 |
| aids700nef | 10 | 64 | 1.0 | 0.0 | 99.918 |
| linux | 8912 | 1 | 0.000112 | 0.0 | 99.999 |
| linux | 8912 | 4 | 0.000449 | 0.0 | 99.997 |
| linux | 8912 | 8 | 0.000898 | 0.0 | 99.995 |
| linux | 8912 | 16 | 0.001795 | 0.0 | 99.99 |
| linux | 8912 | 32 | 0.003591 | 0.0 | 99.98 |
| linux | 8912 | 64 | 0.007181 | 0.0 | 99.96 |
| imdbmulti | 21353 | 1 | 4.7e-05 | 0.0 | 100.0 |
| imdbmulti | 21353 | 4 | 0.000187 | 0.0 | 99.999 |
| imdbmulti | 21353 | 8 | 0.000375 | 0.0 | 99.998 |
| imdbmulti | 21353 | 16 | 0.000749 | 0.0 | 99.996 |
| imdbmulti | 21353 | 32 | 0.001499 | 0.0 | 99.991 |
| imdbmulti | 21353 | 64 | 0.002997 | 0.0 | 99.982 |
| ptc | 11 | 1 | 0.090909 | 0.0 | 99.995 |
| ptc | 11 | 4 | 0.181818 | 0.0 | 99.979 |
| ptc | 11 | 8 | 0.272727 | 0.0 | 99.958 |
| ptc | 11 | 16 | 0.272727 | 0.0 | 99.916 |
| ptc | 11 | 32 | 0.363636 | 0.0 | 99.831 |
| ptc | 11 | 64 | 0.454545 | 0.0 | 99.663 |

### Checkpoint-backed reranking

No current tie-aware GNN reranking run is included. Do not quote the legacy reranking metrics.

## Adapter ablations

| Dataset | Ablation | Setting A | Setting B | MAE A | MAE B |
|---|---|---|---|---:|---:|
| aids700nef | SEGMN projection | all pairs | unprojected | 1.56415 | 1.56415 |
| aids700nef | Graph2Region correction | corrected | original equation | 2.42162 | 3.194981 |
| linux | SEGMN projection | all pairs | unprojected | 0.573866 | 0.573866 |
| linux | Graph2Region correction | corrected | original equation | 1.360405 | 1.360405 |

## Grouped split

Pair-disjoint validation can share graphs. Subject-disjoint validation does not.

| Strategy | Graph overlap | Pair overlap |
|---|---:|---:|
| pair-disjoint | 11 | 0 |
| subject-disjoint | 0 | 0 |

## Checkpoint protocol

Verified `35/35`.
HPO-to-checkpoint hash bindings: `5/35`.

## Still not a paper reproduction

- The current matrix uses one evaluation seed; between-seed variation is not estimated.
- All model-dataset HPO selections are tracked, but only the completed final trainings are hash-bound to active checkpoints.
- Full-corpus evaluation was not run.
