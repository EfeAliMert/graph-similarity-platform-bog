# Project Status

What currently works vs what is still a short local run.

## Working

- Flask UI: select graphs, preview, upload, train, infer, evaluate, retrieve.
- Adapters for SimGNN, GraphSim, SEGMN, Graph Fusion, Graph2Region.
- Training plans for built-in and uploaded GEXF sets.
- Checkpoint paths, sidecar metadata, checksums.
- Unordered-pair holdout (no `(A,B)` / `(B,A)` split leak).
- Optuna budgets: smoke, quick, standard, research.
- Offline 50-pair eval on AIDS / LINUX / IMDB / PTC (`make matrix-existing`).
- Tie-aware retrieval implementation, SEGMN/G2R ablations, grouped-split check.

Results: `reports/RESEARCH_RESULTS.md`. Weights stay local, not in Git.

## Caveats

- Checkpoints are ours, not author releases. Mostly seed 379; some LINUX rows
  use 3407.
- Git tracks all 35 dataset/model HPO selection files. Only the five completed
  AIDS700nef final trainings are hash-bound to active checkpoints; the other
  30 selections have `final_training: not_started`.
- Five-model retrieval mean is just an ablation, not a probability.
- SEGMN did not truncate any of the 50 AIDS/LINUX pairs. G2R correction helped
  AIDS, not LINUX.
- The previous retrieval report used identifier-dependent top-k tie breaking
  and is excluded from current claims. Run `make study` to create a tie-aware
  retrieval report before quoting retrieval numbers.

## Not done

- Paper accuracy tables.
- Seeds 2026 and 3407 on every row.
- Full test-corpus eval.
- A current tie-aware retrieval study.
- MCS / classification from the original papers.

Do not mix exact GED, approximate bounds, and proxy distances in one number.
`Executed` is not an accuracy claim.
