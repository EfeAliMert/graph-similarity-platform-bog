# Graph Similarity Platform

Local Flask app for five graph-similarity models (SimGNN, GraphSim, SEGMN,
Graph Fusion, Graph2Region). The original repos stay separate; this project
converts pairs, loads checkpoints, and reports scores on the same scale where
that conversion is valid.

The repository is ready for reproducible local platform experiments. The
checked-in results are preliminary: they use 50 held-out pairs and one
evaluation seed, so they must not be presented as paper-result reproduction.

```bash
make setup
make models
make datasets
make checkpoints
make run
```

Open http://127.0.0.1:5002 (needs the Flask server, not `file://` on the
template). Then `make check` / `make test`.

`make test` is safe in a fresh clone and reports tests that need ignored model
artifacts as skipped. Use `make test-artifacts` when the complete local model
and checkpoint bundle is installed; that command fails if an artifact is
missing.

Third-party model code, dataset bytes, and weights are not stored in Git.
`make models` clones all five model repositories at the pinned commits and
applies the narrow compatibility patches tracked by this project. `make
datasets` downloads the registered datasets and verifies every GED benchmark
file against `configs/dataset_sources.json`; it also prepares the derived GEXF
layout required by the GraphSim adapter. `make checkpoints` restores the
35 locally trained model/dataset checkpoints from the versioned GitHub Release
and verifies the archive and every extracted file. These are project weights,
not author-released pretrained weights. See `docs/ARTIFACT_SETUP.md` for the
exact source identities and artifact boundaries.

```bash
make models-verify
make datasets-verify
make checkpoints-verify
make checkpoint-audit
make preflight
```

## Models

| Model | Local code | Notes |
| --- | --- | --- |
| SimGNN | `SimGNN-v_00001` | PyTorch adapter |
| GraphSim | `GraphSim-master` | TF1 + validation isotonic calibration |
| SEGMN | `SEGMN-main` | assignment-graph input |
| Graph Fusion | `GFM-code` | GEXF / degree features |
| Graph2Region | `Graph2Region-main` | PyG + stored GED-volume correction |

`Executed` only means a local checkpoint ran. It is not a paper-table result.

## Datasets

| Dataset | Target here |
| --- | --- |
| AIDS700nef, LINUX | exact A* GED |
| IMDBMulti, PTC | min(Beam, Hungarian, VJ) upper bound |
| MUTAG, PROTEINS, ENZYMES | structural proxy (not GED) |

Uploaded GED files are tagged `unverified_ged`.

To install only selected datasets:

```bash
.venvs/gnn-pyg/bin/python scripts/fetch_datasets.py \
  --datasets aids700nef linux mutag
```

The downloader fetches benchmark files directly from the GraphSim authors'
shared folder. MUTAG, PROTEINS, and ENZYMES are exported through PyTorch
Geometric. Upstream dataset terms still apply.

Default comparable score is `exp(-GED / mean(|V|))`. GraphSim uses
`exp(-0.7 * d_norm)` and must be inverted through its checkpoint calibrator.

## Training / HPO

```bash
.venvs/gnn-pyg/bin/python scripts/optimize.py \
  --model simgnn --dataset aids700nef --budget standard --seed 379
make hpo
```

HPO uses validation similarity MSE. Pair key is
`(min(id_a, id_b), max(id_a, id_b))`. Details: `docs/HPO_ARCHITECTURE.md`.
All 35 dataset/model selections are tracked. Five AIDS700nef final checkpoints
are currently hash-bound to those selections; the other active checkpoints
have protocol metadata but are not claimed as HPO-produced weights.

## Evaluation

```bash
make matrix-existing
make study
```

50-pair held-out numbers: `reports/RESEARCH_RESULTS.md`. One eval seed, not
the 3-seed protocol in `docs/RESEARCH_PROTOCOL.md`. Single-seed rows report a
pair-bootstrap 95% CI; between-seed standard deviation is shown as unavailable.

Retrieval: structural prefilter, then GNN rerank (budget capped at 100).
Top-k relevance includes every pair tied at the GED cutoff.

Upload a GEXF zip from the UI, or see `docs/ADDING_NEW_DATASETS.md`.

```
graph_similarity_platform/   app, adapters, eval, HPO
templates/, static/          UI
scripts/                     train / optimize / audit
configs/, tests/, docs/, reports/
```

MIT for this wrapper. Third-party model code and data keep their own licenses.
