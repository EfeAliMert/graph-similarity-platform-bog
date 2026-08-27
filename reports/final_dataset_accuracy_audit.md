# Dataset Accuracy Audit

This is a local-checkpoint study, not a reproduction of author-released paper results.

Matrix complete: `True`. Mode: `evaluate-existing`. Held-out pairs: `50`.

## aids700nef

Reference: **exact**.

| Model | Checkpoint seeds | GED MAE | Pair-bootstrap 95% CI | MSE x1e3 | Spearman | NDCG@k | Split verified |
|---|---|---:|---|---:|---:|---:|---|
| Graph Fusion | 379 | 1.211 | [0.886, 1.500] | 4.794 | 0.819 | 0.933 | True |
| Graph2Region | 379 | 2.422 | [1.827, 3.102] | 21.924 | 0.511 | 0.815 | True |
| Multi-Scale Convolutional Set Matching | 379 | 2.413 | [1.928, 3.070] | 18.419 | 0.586 | 0.897 | True |
| SEGMN | 379 | 1.564 | [1.076, 2.203] | 4.906 | 0.750 | 0.922 | True |
| SimGNN | 379 | 1.831 | [1.431, 2.246] | 13.133 | 0.705 | 0.904 | True |

## imdbmulti

Reference: **approximate_benchmark**.

| Model | Checkpoint seeds | GED MAE | Pair-bootstrap 95% CI | MSE x1e3 | Spearman | NDCG@k | Split verified |
|---|---|---:|---|---:|---:|---:|---|
| Graph Fusion | 379 | 70.391 | [26.140, 131.904] | 22.000 | 0.825 | 0.915 | True |
| Graph2Region | 379 | 80.125 | [34.620, 147.206] | 14.976 | 0.913 | 0.925 | True |
| Multi-Scale Convolutional Set Matching | 379 | 68.412 | [28.807, 133.103] | 13.103 | 0.882 | 0.966 | True |
| SEGMN | 379 | 54.471 | [24.221, 102.805] | 8.803 | 0.899 | 0.965 | True |
| SimGNN | 379 | 81.088 | [42.090, 137.753] | 87.590 | 0.936 | 0.918 | True |

## linux

Reference: **exact**.

| Model | Checkpoint seeds | GED MAE | Pair-bootstrap 95% CI | MSE x1e3 | Spearman | NDCG@k | Split verified |
|---|---|---:|---|---:|---:|---:|---|
| Graph Fusion | 3407 | 0.805 | [0.578, 1.061] | 4.452 | 0.936 | 0.996 | True |
| Graph2Region | 379 | 1.360 | [1.058, 1.719] | 10.610 | 0.821 | 0.953 | True |
| Multi-Scale Convolutional Set Matching | 3407 | 1.084 | [0.876, 1.299] | 11.013 | 0.908 | 0.935 | True |
| SEGMN | 379 | 0.574 | [0.452, 0.714] | 2.252 | 0.966 | 0.989 | True |
| SimGNN | 3407 | 1.007 | [0.784, 1.242] | 11.412 | 0.915 | 0.776 | True |

## ptc

Reference: **approximate_benchmark**.

| Model | Checkpoint seeds | GED MAE | Pair-bootstrap 95% CI | MSE x1e3 | Spearman | NDCG@k | Split verified |
|---|---|---:|---|---:|---:|---:|---|
| Graph Fusion | 379 | 17.114 | [7.565, 32.350] | 8.960 | 0.935 | 0.936 | True |
| Graph2Region | 379 | 21.335 | [15.989, 27.670] | 31.182 | 0.560 | 0.615 | True |
| Multi-Scale Convolutional Set Matching | 379 | 8.013 | [6.249, 10.352] | 11.186 | 0.950 | 0.879 | True |
| SEGMN | 379 | 14.838 | [9.365, 21.618] | 19.463 | 0.801 | 0.890 | True |
| SimGNN | 379 | 42.701 | [16.434, 82.006] | 10.590 | 0.933 | 0.959 | True |
