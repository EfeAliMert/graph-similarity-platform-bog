# Research Protocol

Fixed experiment rules. Generated reports should not rewrite these.

## Question

Does structural prefilter + checkpoint GNN beat structural ranking alone, at
a fixed candidate budget?

Expect: recall improves with budget then flattens; GNN NDCG should beat the
structural order at the same budget; matching models may break when test
graphs are larger than training graphs. Those are hypotheses.

## Data

- AIDS700nef, LINUX: local A* GED maps.
- IMDBMulti, PTC: min(Beam, Hungarian, VJ); approximate, not exact GED.
- MUTAG, PROTEINS, ENZYMES: structural proxy, separate tables.
- Uploaded GED stays `unverified_ged` until solver/costs are known.

Keep the original train/test graph split. Validation holdout is from training
pairs only.

## Splits

Pair key: `(min(id_a, id_b), max(id_a, id_b))`. If a pair is in validation,
neither direction may appear in training. One split per dataset/model study.

Pair-disjoint splits can still share graphs. Grouped data needs a subject- or
graph-disjoint split; record seed, counts, overlap, SHA-256.

## Training

Seeds 379, 2026, 3407. Pick checkpoints by validation similarity MSE. No test
pairs in HPO. Report mean ± std only when at least two seeds completed; show
standard deviation as unavailable for a single seed. Keep failures in the log.
Store hyperparameters, feature mode, projection limits, runtime, and target
equation with the run. Early stop where the trainer supports it.

## Metrics

GED MAE/RMSE, normalized-GED MAE, similarity MAE/RMSE/MSE, Spearman, Kendall
tau-b, P/R/NDCG@k. Latency p50/p95, retrieval time, throughput, peak memory.
Uncertainty: pair bootstrap 95% CI within each evaluated pair set and seed std
across independent seeds. These answer different questions and are not pooled.

Do not mix exact, approximate, unverified, and proxy targets in one aggregate.

## Retrieval

Budgets `1, 4, 8, 16, 32, 64`. Record candidate recall@k, best-pair recovery,
best reference distance in the set, GED regret, reranked NDCG@k, time.
When multiple pairs share the GED at rank k, every pair at that cutoff is
relevant. Recall uses this expanded set, so graph identifier order cannot alter
the result. Precision still uses the k returned pairs.

Baselines: structural order, each GNN, optional five-model mean of canonical
similarities (only if all five adapters run). GraphSim without its validation
calibrator is reported unavailable, not clipped.

```bash
make matrix-existing
make study
```

`make matrix-existing`: 50 stratified pairs, stored checkpoints.
`make study`: also retrieval, adapter ablations, grouped split, compiled report.
3-seed retrain: `make matrix-full`.

`Executed` means one local forward pass completed, not a paper reproduction.
