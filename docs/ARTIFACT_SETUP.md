# Local Artifact Setup

The Git repository contains the Flask application, adapters, training scripts,
tests, and documentation. It does not contain the five model repositories,
dataset archives, Python environments, or trained checkpoints.

This guide explains what has to be added for real inference. A source-only
clone can start the site and run the normal tests, but model cards cannot show
`Executed` until the matching local artifacts exist.

## What Is Missing from Git

These paths are intentionally ignored:

```text
Models&Datasets/
.venvs/
training_logs/
output/
tmp/
```

They are excluded because they are large and because third-party code, data,
and derived checkpoints may have different distribution terms. Check
[`THIRD_PARTY.md`](THIRD_PARTY.md) before sharing any of them.

## Tested Local Environment

The development workspace used:

```text
Python 3.9.6
torch 2.1.1
torch-geometric 2.4.0
TensorFlow 2.15.1 through tensorflow.compat.v1
```

The complete package pins are in:

```text
requirements.txt
requirements-gnn-pyg.txt
requirements-graphsim.txt
```

The base CI workflow also checks the source code with Python 3.11. That does
not mean every legacy model dependency has been tested on Python 3.11.

## 1. Create the Environments

From the repository root:

```bash
make setup
```

This installs the Flask dependencies and creates:

```text
.venvs/gnn-pyg/     SimGNN, SEGMN, Graph Fusion, Graph2Region, and HPO
.venvs/graphsim/    GraphSim and TensorFlow compatibility code
```

Check both environments before downloading or training anything large:

```bash
.venvs/gnn-pyg/bin/python -m pip check
.venvs/graphsim/bin/python -m pip check
.venvs/gnn-pyg/bin/python -c 'import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__)'
.venvs/graphsim/bin/python -c 'import tensorflow as tf; print(tf.__version__)'
```

`make setup` installs Python packages only. It does not obtain model source
code, datasets, or weights.

## 2. Add the Model Source Folders

The adapter registry expects this exact layout:

```text
Models&Datasets/
  SimGNN-v_00001/
  GraphSim-master/
  SEGMN-main/
  GFM-code/
  Graph2Region-main/
```

The sources recorded in the local inventory are:

| Model | Upstream or paper link | Local source identity |
| --- | --- | --- |
| SimGNN | <https://github.com/benedekrozemberczki/SimGNN> | Extracted local snapshot; Git commit was not recoverable |
| GraphSim | <https://github.com/yunshengb/GraphSim> | Extracted local snapshot with TensorFlow/networkx compatibility edits; Git commit was not recoverable |
| SEGMN | <https://arxiv.org/abs/2411.03624> | Supplied local archive; no verifiable Git remote was found |
| Graph Fusion | <https://github.com/LLiRarry/GFM-code> | Commit `a10c8870b79351963b61f9f3c113e2545ebdc23d` |
| Graph2Region | <https://github.com/liuzhouyang/Graph2Region> | Extracted local snapshot plus `dataset_g2r.py` compatibility shim; Git commit was not recoverable |

The public repositories can be placed at the required paths with:

```bash
mkdir -p 'Models&Datasets'
git clone https://github.com/benedekrozemberczki/SimGNN.git 'Models&Datasets/SimGNN-v_00001'
git clone https://github.com/yunshengb/GraphSim.git 'Models&Datasets/GraphSim-master'
git clone https://github.com/LLiRarry/GFM-code.git 'Models&Datasets/GFM-code'
git clone https://github.com/liuzhouyang/Graph2Region.git 'Models&Datasets/Graph2Region-main'
git -C 'Models&Datasets/GFM-code' checkout a10c8870b79351963b61f9f3c113e2545ebdc23d
```

There is no clone command for SEGMN because the development archive had no
verified repository URL. Place the reviewed source snapshot at
`Models&Datasets/SEGMN-main`.

Cloning the current SimGNN, GraphSim, or Graph2Region default branch does not
reproduce the exact development snapshot. GraphSim needs the narrow legacy
compatibility edits used by the local runner, and Graph2Region needs
`Models&Datasets/Graph2Region-main/dataset_g2r.py`. Until those snapshots or
patches are published under suitable terms, they must be transferred from the
audited local artifact set. This is a known reproducibility limit, not a setup
step that should be guessed.

Check the main entrypoints:

```bash
test -f 'Models&Datasets/SimGNN-v_00001/src/main.py'
test -f 'Models&Datasets/GraphSim-master/model/Siamese/run.py'
test -f 'Models&Datasets/SEGMN-main/main.py'
test -f 'Models&Datasets/GFM-code/model/Regression.py'
test -f 'Models&Datasets/Graph2Region-main/run.py'
test -f 'Models&Datasets/Graph2Region-main/dataset_g2r.py'
```

## 3. Install the Datasets

After `make setup`, install every registered dataset with:

```bash
make datasets
```

This command downloads AIDS700nef, LINUX, IMDBMulti, and PTC from the
[GraphSim authors' shared benchmark folder](https://drive.google.com/drive/folders/1JcAgWKYC41687UeiLaFg-QlPmIpZvWhT?usp=sharing).
It verifies each archive and GED/MCS map against the byte count and SHA-256 in
`configs/dataset_sources.json` before moving the file into place. MUTAG,
PROTEINS, and ENZYMES are downloaded and exported through PyTorch Geometric.

To install a subset:

```bash
.venvs/gnn-pyg/bin/python scripts/fetch_datasets.py \
  --datasets aids700nef linux mutag
```

To check an existing installation without network access:

```bash
make datasets-verify
```

The built-in registry reads files from:

```text
Models&Datasets/drive-download-20260630T100606Z-3-001/
```

Expected graph archives after installation:

```text
AIDS700nef.zip
LINUX.tar.gz
IMDBMulti.zip
PTC.zip
MUTAG.zip
PROTEINS.zip
ENZYMES.zip
```

The TU exports also include `mutag_graph_labels.json`,
`proteins_graph_labels.json`, and `enzymes_graph_labels.json`.

Expected GED/MCS maps for the four GED benchmark collections:

```text
aids700nef_ged_astar_gidpair_dist_map.pickle
aids700nef_mcs_mccreesh2017_gidpair_dist_map.pickle
linux_ged_astar_gidpair_dist_map.pickle
linux_mcs_mccreesh2017_gidpair_dist_map.pickle
imdbmulti_ged_astar_gidpair_dist_map.pickle
imdbmulti_mcs_mccreesh2017_gidpair_dist_map.pickle
ptc_ged_astar_gidpair_dist_map.pickle
ptc_mcs_mccreesh2017_gidpair_dist_map.pickle
```

The registry treats AIDS700nef and LINUX as exact A* GED. IMDBMulti and PTC
remain approximate GED benchmark upper bounds even though their filenames
contain `astar`.

These three datasets do not provide pairwise GED labels in this project. Their
training target is the documented structural proxy.

Another dataset can be uploaded instead of restoring the built-in archives.
The accepted layout and target rules are in
[`ADDING_NEW_DATASETS.md`](ADDING_NEW_DATASETS.md).

## 4. Verify Dataset Checksums

The source manifest is the stable download allowlist and checksum record:

```bash
make datasets-verify
```

`configs/dataset_sources.json` is reviewed and committed independently of the
local data. `artifacts.manifest.json` records the files present in one audited
workspace, but it is not used as the download trust root.

The expected graph counts are:

| Dataset | Train | Test | Total | Target used locally |
| --- | ---: | ---: | ---: | --- |
| AIDS700nef | 560 | 140 | 700 | Exact A* GED |
| LINUX | 800 | 200 | 1,000 | Exact A* GED |
| IMDBMulti | 1,200 | 300 | 1,500 | Approximate GED upper bound |
| PTC | 275 | 69 | 344 | Approximate GED upper bound |
| MUTAG | 150 | 38 | 188 | Structural proxy |
| PROTEINS | 890 | 223 | 1,113 | Structural proxy |
| ENZYMES | 480 | 120 | 600 | Structural proxy |

## 5. Create or Restore Checkpoints

No author-released pretrained weights are bundled with this project. The
development checkpoints were trained locally and selected by validation
similarity MSE.

Expected paths follow this pattern:

| Model | Checkpoint path |
| --- | --- |
| SimGNN | `Models&Datasets/SimGNN-v_00001/checkpoints/simgnn_<dataset>.pt` |
| GraphSim | `Models&Datasets/GraphSim-master/checkpoints/<dataset>/graphsim.ckpt*` |
| SEGMN | `Models&Datasets/SEGMN-main/checkpoints/<dataset>/segmn_<dataset>_best.pt` |
| Graph Fusion | `Models&Datasets/GFM-code/checkpoints/gfm_<dataset>.pt` |
| Graph2Region | `Models&Datasets/Graph2Region-main/checkpoints/<dataset>/g2r_<dataset>_best.pt` |

There are two valid routes:

1. Restore the audited local checkpoint files and their metadata sidecars.
2. Train new dataset-specific checkpoints and report them as new runs.

For new training, start the site:

```bash
make run
```

Open `http://127.0.0.1:5002`, select a dataset, and use the training section.
The same plans can be inspected through:

```bash
curl 'http://127.0.0.1:5002/api/training?dataset=aids700nef'
```

Each row must show `can_start: true`. The web job records its command,
checkpoint path, seed, target source, and log. Direct commands for all five
trainers are documented in [`ADDING_NEW_DATASETS.md`](ADDING_NEW_DATASETS.md).

For HPO:

```bash
.venvs/gnn-pyg/bin/python scripts/optimize.py \
  --model all \
  --dataset aids700nef \
  --budget standard \
  --seed 379
```

HPO selects settings using validation data. It does not turn its trial weights
into final test evidence. The normal training workflow must create the final
checkpoint afterward.

## 6. Run the Audits

Start with source and environment checks:

```bash
make check
make test
```

`make test` runs from `.venvs/gnn-pyg`. Tests that require ignored model
sources or checkpoints are reported as skipped when those artifacts are not
installed. To require the complete local artifact bundle and fail on anything
missing, run:

```bash
make test-artifacts
```

Then inspect the model artifacts:

```bash
make checkpoint-audit
make model-audit
make manifest
```

`make manifest` runs with `.venvs/gnn-pyg`, refreshes the checkpoint audit,
and hashes the resulting evidence files. Do not edit an audited file after this
step without regenerating the manifest.

Start Flask before the HTTP preflight:

```bash
make run
```

In a second terminal:

```bash
make preflight
```

Only regenerate the inventory after the restored or newly trained artifacts
have passed review:

```bash
make manifest
```

The following meanings stay separate:

- `Runnable`: required code and Python dependencies were found.
- `Checkpoint available`: a file exists at the registered dataset path.
- `Protocol verified`: metadata passed the local provenance and split audit.
- `Executed`: a forward pass completed for the selected graph pair.
- Accurate: held-out metrics support the claim under the declared protocol.

The first four states do not prove the fifth.

## 7. Fresh-Clone Expectations

| Workspace state | What should work |
| --- | --- |
| Source only | Flask startup, interface, syntax checks, and base unit tests |
| Environments added | Dependency checks and HPO/training-plan inspection |
| Model sources added | Adapter discovery; checkpoints may still be missing |
| Datasets added | Graph catalog, pair preview, target inspection, and training plans |
| Checkpoints added or trained | Real pair inference and checkpoint-backed retrieval |
| Audited checkpoints plus held-out runs | Research tables within the stated target and split limits |

If a source-only clone shows `Runtime missing`, `Repository missing`, or
`Checkpoint required`, that is expected. The platform should report the
missing artifact instead of fabricating a model score.

## Current Reproducibility Limit

The platform code and Python requirements are versioned, but four third-party
source snapshots do not have recoverable Git commits in the local manifest.
SEGMN also lacks a verified public code remote. Therefore the current repository
supports an audited local reconstruction, not a fully automatic download of
the exact five-source snapshot.

Closing this gap requires license-compatible patch files or source snapshot
hashes for SimGNN, GraphSim, SEGMN, and Graph2Region. Until then, keep the
artifact provenance visible in any shared result.
