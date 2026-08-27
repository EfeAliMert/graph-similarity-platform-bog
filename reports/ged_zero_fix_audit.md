# GED=0 Checkpoint Audit

Dataset: `AIDS700nef`

Audited pair: `AIDS700nef/train/99.gexf` vs
`AIDS700nef/test/27325.gexf`

Reference: exact A* GED `0`, canonical similarity `1.0`.

## Root causes

- The normalized-GED round-robin holdout could move all 23 distinct GED=0
  training pairs into validation.
- GraphSim read an unavailable `global_id` field in inductive mode and therefore
  failed to identify distinct zero-GED pairs for oversampling.
- SEGMN's similarity-matrix CNN was sensitive to arbitrary node order in the
  universal adapter.
- Graph Fusion did not receive identity anchors during local training.
- The supplied Graph2Region code generated fresh random positional indices for
  each forward pass and its GED-volume expression did not map identical regions
  to zero distance.

## Active pair results

| Model | Comparable similarity | Predicted GED | Identity similarity |
|---|---:|---:|---:|
| SimGNN | 0.940208 | 0.616538 | 0.940208 |
| GraphSim | 0.533713 | 6.278961 | 0.538460 |
| SEGMN | 0.998565 | 0.014360 | 0.998565 |
| Graph Fusion | 0.618741 | 4.800692 | 0.863478 |
| Graph2Region | 1.000000 | 0.000000 | 1.000000 |

The table reports learned outputs without an exact-isomorphism override.
GraphSim and Graph Fusion remain inaccurate on this pair. Their zero-balanced
candidates were rejected because they worsened held-out GED or ranking metrics.

## Active 24-pair AIDS benchmark

| Model | GED MAE | Similarity MSE | Spearman rho | NDCG@5 |
|---|---:|---:|---:|---:|
| SimGNN | 1.238662 | 0.008948 | 0.871000 | 0.986796 |
| GraphSim | 2.797445 | 0.023265 | 0.792137 | 0.829335 |
| SEGMN | 1.445667 | 0.004420 | 0.863552 | 0.993086 |
| Graph Fusion | 1.800156 | 0.012325 | 0.724665 | 0.976164 |
| Graph2Region | 2.181452 | 0.010571 | 0.509106 | 0.932196 |

Artifact: `training_logs/benchmarks/20260823T095147Z-d63b8bde.json`.

Verification: 62 unit/integration tests passed; checkpoint protocol audit
verified 35/35 model-dataset entries; live `/api/compare` executed all five
checkpoint-backed architectures with verified input binding.
