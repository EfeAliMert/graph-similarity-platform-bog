# Adding a New Dataset

Convert graphs to GEXF, upload one archive, check the manifest, then train a
checkpoint per model. Upload does not need a code change. Static registration
is only for a reviewed built-in benchmark.

## Before Converting Anything

First decide what one graph represents and what a pair target means. This is
more important than the file format. For example, a molecule graph, a source
code graph, and a social-network subgraph need different node labels, split
rules, and edit costs.

Record at least:

- Source URL and dataset version.
- Meaning of nodes, edges, and node features.
- Directed or undirected graph convention.
- Graph construction and preprocessing steps.
- Split unit, such as molecule, subject, project, or time period.
- Pair-target source and edit-cost convention.
- Converter version and random seed.

Changing one of these choices can change the learning problem. In that case I
use a new dataset ID instead of silently replacing the old files.

## Archive Layout

Each graph must be one `.gexf` file. Accepted archives are `.zip`, `.tar`,
`.tar.gz`, and `.tgz`.

The preferred layout is:

```text
my-dataset/
  train/
    0.gexf
    1.gexf
  test/
    100.gexf
    101.gexf
  ged.csv                 # optional
```

The outer folder is optional. The `train/` and `test/` path segments are what
matter.

If both split folders exist, the importer keeps them. If they do not, it sorts
all graph paths and creates a deterministic split of about 80% train and 20%
test, with at least one graph on each side. For a real experiment, create the
split before upload; the importer does not know subject, project, scaffold, or
time-group identities.

All GEXF filename stems must be unique across the archive. This is invalid:

```text
train/subject.gexf
test/subject.gexf
```

Numeric stems are preserved as graph IDs. Otherwise, sorted graph paths receive
IDs from `0` to `N-1`. The mapping is saved in `manifest.json`.

## GEXF Fields Used by Training

A graph must contain at least one node. The universal training parser uses:

- Node IDs.
- Edges between different nodes.
- The node attribute titled `type`, when present.
- Otherwise the first available node attribute, node label, or `0`.
- Numeric node attributes named `feature_0`, `feature_1`, and so on, when every
  node has the same feature fields.

The current universal adapters do not use edge attributes. Self-loops are
removed by the training conversion. Keep these limits in mind when converting
an attributed or directed dataset; storing information in GEXF does not mean
that every model consumes it.

A small labeled graph can be written as:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">
  <graph mode="static" defaultedgetype="undirected">
    <attributes class="node">
      <attribute id="type" title="type" type="string" />
    </attributes>
    <nodes>
      <node id="0" label="C">
        <attvalues><attvalue for="type" value="C" /></attvalues>
      </node>
      <node id="1" label="O">
        <attvalues><attvalue for="type" value="O" /></attvalues>
      </node>
    </nodes>
    <edges>
      <edge id="0" source="0" target="1" />
    </edges>
  </graph>
</gexf>
```

Use one dataset-level feature rule. Normalizing each graph independently can
remove differences that the similarity model is supposed to learn.

## Target Types

The registry separates four target kinds:

| Kind | Meaning | Suitable claim |
| --- | --- | --- |
| `exact` | Distance from a verified exact solver with documented edit costs | Exact-GED evaluation |
| `approximate_benchmark` | Published or reproducible approximate-solver upper bound | Approximate-reference evaluation |
| `unverified_ged` | User-uploaded distances whose solver provenance was not checked by the site | Agreement with the supplied labels |
| `structural_proxy` | Deterministic descriptor distance used when GED coverage is unavailable | Pipeline and proxy-ranking experiments |

Every GED file uploaded through the web interface is marked
`unverified_ged`. A filename containing `astar` is not proof of exactness. An
exact claim also needs the solver, version, edit costs, preprocessing, timeout
policy, and completion evidence.

When usable GED coverage is absent, training uses this structural proxy:

```text
|node count difference|
+ |edge count difference|
+ 0.5 * L1(node-label histograms)
+ 0.5 * L1(degree histograms)
```

This value is not GED and must not be shown in an exact-GED table.

## GED File Format

GED values may be placed inside the archive or uploaded separately. An
embedded file is detected when its filename contains `ged` and ends in `.csv`
or `.json`.

CSV with a header:

```csv
left,right,ged
0,1,2
0,100,4
```

The left column may be named `left`, `graph_a`, `graph1`, or `left_id`. The
right side accepts the corresponding right-side names. The distance column may
be `ged`, `distance`, or `edit_distance`. A headerless three-column CSV also
works.

JSON object:

```json
{
  "0,1": 2,
  "0,100": 4
}
```

JSON rows:

```json
[
  {"left": 0, "right": 1, "ged": 2},
  {"graph_a": "train/0.gexf", "graph_b": "test/100.gexf", "distance": 4}
]
```

All distances must be finite and non-negative, and every graph reference must
exist. The importer accepts assigned IDs, original stems, filenames, or full
archive paths.

Pairs are symmetric. `(a,b)` and `(b,a)` are stored under one unordered key. If
two approximate directional records disagree, the importer keeps the smaller
finite upper bound.

An uploaded GED target is `training_ready` only when it has:

- At least two training graphs.
- At least one test graph.
- At least one train/train GED pair.
- At least one test/train GED pair.

If this minimum is not met, sparse GED values are not mixed with proxy values
during training. The trainer uses the structural proxy instead. The minimum is
enough to test the pipeline, but not enough for a useful accuracy experiment.

## Upload Limits

| Item | Limit |
| --- | ---: |
| Uploaded archive | 512 MB |
| Expanded validated content | 512 MB |
| One GEXF file | 20 MB |
| Graphs in one archive | 20,000 |
| Separate GED file | 512 MB |

The importer rejects unsafe archive paths, duplicate graph stems, empty or
invalid graphs, unknown GED references, negative distances, and non-finite
values.

## Upload from the Site

1. Start Flask with `make run`.
2. Open `http://127.0.0.1:5002`.
3. Choose the graph archive in the dataset section.
4. Enter a name and optional stable ID.
5. Add a GED CSV/JSON file if available.
6. Upload and inspect several train/test graph previews.

Dataset IDs are lowercased and changed to letters, numbers, and hyphens. The
upload endpoint does not overwrite an existing ID.

## Upload from the API

```bash
curl -X POST http://127.0.0.1:5002/api/datasets/upload \
  -F 'name=My Graph Dataset' \
  -F 'dataset_id=my-graph-dataset' \
  -F 'domain=Molecular graphs' \
  -F 'archive=@/absolute/path/my-dataset.zip' \
  -F 'ground_truth=@/absolute/path/ged.csv'
```

Leave out `ground_truth` when no pair target exists.

Check the result:

```bash
curl http://127.0.0.1:5002/api/datasets
curl http://127.0.0.1:5002/api/datasets/my-graph-dataset/graphs
curl http://127.0.0.1:5002/api/datasets/my-graph-dataset
```

The normalized local files are stored at:

```text
Models&Datasets/uploaded_datasets/my-graph-dataset/
  dataset.zip
  ged.json          # only when GED was accepted
  manifest.json
```

These files are ignored by Git.

## Check the Manifest

```bash
python3 -m json.tool \
  'Models&Datasets/uploaded_datasets/my-graph-dataset/manifest.json'
```

The important fields are the target kind, `training_ready`, graph counts,
training/test pair counts, and `graph_name_map`. Do not renumber graphs after
training. IDs are part of pair binding, dataset fingerprints, checkpoints, and
saved evaluations.

## Prepare the Model Runtimes

The adapters expect:

```text
Models&Datasets/
  SimGNN-v_00001/
  GraphSim-master/
  SEGMN-main/
  GFM-code/
  Graph2Region-main/
```

Review [`THIRD_PARTY.md`](THIRD_PARTY.md) before copying or publishing these
repositories. Then create the environments:

```bash
make setup
make models
make models-verify
.venvs/gnn-pyg/bin/python -m pip check
.venvs/graphsim/bin/python -m pip check
```

`make setup` installs Python packages. `make models` downloads the five pinned
third-party repositories and applies the platform compatibility patches.

With Flask running, inspect the five generated training plans:

```bash
curl 'http://127.0.0.1:5002/api/training?dataset=my-graph-dataset'
```

Each plan should have `can_start: true`. If it does not, the `detail` field
normally identifies a missing model folder, environment, package, or invalid
dataset manifest.

The model IDs are:

| Model | ID |
| --- | --- |
| SimGNN | `simgnn` |
| GraphSim | `multiscale-set` |
| SEGMN | `segmn` |
| Graph Fusion | `graph-fusion` |
| Graph2Region | `graph2region` |

## Train a Checkpoint

Training from the site and training through the API use the same job service.
A normal API request looks like this:

```bash
curl -X POST http://127.0.0.1:5002/api/training/start \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "simgnn",
    "dataset_id": "my-graph-dataset",
    "epochs": 25,
    "batch_size": 32,
    "seed": 379,
    "optimize": false
  }'
```

Repeat the request with the other model IDs. On a memory-limited computer I
run them one at a time. SEGMN's assignment graph and Graph Fusion's attention
can use much more memory than a simple node count suggests.

Expected checkpoint locations are:

| Model | Path |
| --- | --- |
| SimGNN | `Models&Datasets/SimGNN-v_00001/checkpoints/simgnn_<dataset>.pt` |
| GraphSim | `Models&Datasets/GraphSim-master/checkpoints/<dataset>/graphsim.ckpt*` |
| SEGMN | `Models&Datasets/SEGMN-main/checkpoints/<dataset>/segmn_<dataset>_best.pt` |
| Graph Fusion | `Models&Datasets/GFM-code/checkpoints/gfm_<dataset>.pt` |
| Graph2Region | `Models&Datasets/Graph2Region-main/checkpoints/<dataset>/g2r_<dataset>_best.pt` |

The generated training command and log path are returned by the API and saved
with the job. Use that instead of reconstructing the command later.

## Run HPO

First make sure the default training path finishes. HPO then tries real
settings connected to each model's training command and minimizes validation
similarity MSE.

```bash
.venvs/gnn-pyg/bin/python scripts/optimize.py \
  --model all \
  --dataset my-graph-dataset \
  --budget standard \
  --seed 379
```

Budgets are defined in `configs/hpo/budgets.json`:

| Budget | Trials | Confirmation seeds |
| --- | ---: | --- |
| `smoke` | 2 | 379 |
| `quick` | 8 | 379 |
| `standard` | 24 | 379, 2026, 3407 |
| `research` | 50 | 379, 2026, 3407 |

The selected file is saved at
`configs/optimized/<dataset>/<model>.json`. It is ignored when the dataset
fingerprint or search-space version changes. A trial checkpoint is not final
test evidence; final training happens after configuration selection.

See [`HPO_ARCHITECTURE.md`](HPO_ARCHITECTURE.md) for the exact parameters and
study rules.

## Checkpoint Metadata

For a research result, the checkpoint or sidecar should record:

- Model, dataset, architecture class, and code origin.
- Dataset fingerprint and target provenance.
- Native target equation and feature mode.
- Training/validation pair counts and split seed.
- Pair overlap and any required graph/subject overlap.
- Split SHA-256 identifier.
- Optimizer, learning rate, batch size, and training budget.
- Checkpoint-selection rule and best validation value.
- Runtime/package versions and weight checksum.
- GraphSim calibration provenance when applicable.

`Executed` only confirms a completed forward pass with the registered model and
checkpoint.

## Score Meaning

Most models use:

```text
average_size = (|V1| + |V2|) / 2
d_norm = GED / average_size
s = exp(-d_norm)
```

The inverse is:

```text
predicted_GED = -average_size * log(max(predicted_similarity, epsilon))
```

GraphSim uses `exp(-0.7 * d_norm)`, so its inverse includes the `0.7` factor.
Its raw regression output must first pass through the checkpoint's
validation-only calibrator. Invalid or missing calibration is reported as
unavailable instead of clipping the output.

After recovering predicted GED, the shared display score is:

```text
comparable_similarity = exp(-predicted_GED / average_size)
```

This conversion is valid only when the checkpoint metadata matches the stated
target. HPO does not make native model outputs directly comparable.

## Evaluate and Retrieve

The web evaluator accepts exact and approximate benchmark references. It keeps
their result tables separate. Uploaded `unverified_ged` labels are usable for
training when coverage is sufficient, but they are not promoted to the
built-in exact/approximate benchmark endpoint without a provenance review.

Example held-out benchmark request:

```bash
curl -X POST \
  http://127.0.0.1:5002/api/datasets/my-graph-dataset/evaluate \
  -H 'Content-Type: application/json' \
  -d '{
    "methods": ["simgnn", "multiscale-set", "segmn", "graph-fusion", "graph2region"],
    "sample_size": 50,
    "sample_mode": "stratified",
    "scope": "train-test",
    "seed": 379,
    "top_k": 10
  }'
```

The synchronous evaluator is capped at 200 pairs. For final work, repeat the
declared protocol over seeds 379, 2026, and 3407 and report mean plus standard
deviation. Do not choose only the best seed.

Checkpoint-backed pair search is capped at 100 candidate pairs:

```bash
curl -X POST \
  http://127.0.0.1:5002/api/datasets/my-graph-dataset/best-pair \
  -H 'Content-Type: application/json' \
  -d '{
    "methods": ["simgnn", "segmn"],
    "max_pairs": 64,
    "scope": "train-test"
  }'
```

Report the search scope, candidate rule, budget, reference kind, and whether
the returned pair was checked against that reference.

## Register a Built-In Dataset

Static registration after checking dataset version and target provenance:

1. Add the archive and target paths to `ORIGINAL_DATASETS` in
   `graph_similarity_platform/data.py`.
2. Add the storage record to `STATIC_DATASETS` in
   `scripts/universal_dataset.py`.
3. Add the ID to `ALL_DATASETS` in
   `graph_similarity_platform/models/real_models.py`.
4. Add a special checkpoint path only if the normal dynamic path is not enough.
5. Add target-audit and API tests.
6. Record the source, version, license, and checksums in the documentation.
7. Run the checkpoint and model-output audits.
8. Regenerate `artifacts.manifest.json`.

Do not set `target_exact: true` based only on a filename or paper description.

## Verification

Run:

```bash
make check
make test
.venvs/gnn-pyg/bin/python -m unittest tests.test_training_batches -v
make checkpoint-audit
make model-audit
make manifest
```

Start Flask before `make preflight`.

Manual checks:

- Source graph count matches the imported count.
- Train/test membership matches the intended split.
- Several previews have plausible nodes, edges, labels, and features.
- GED IDs point to the intended graphs.
- Identity distances are zero when identity pairs are included.
- Pair distances are symmetric under the declared convention.
- No unordered pair crosses train and validation.
- No subject or source object crosses a grouped split.
- Each checkpoint matches the current dataset fingerprint.
- Swapping graph order has either a small error or a documented limitation.
- Missing references remain missing and are not replaced by proxy values.

## Common Problems

| Message or symptom | What to check |
| --- | --- |
| Archive needs at least two GEXF graphs | File extensions and archive contents |
| GEXF stems must be unique | Duplicate names across train/test |
| GED references a missing graph | GED keys against `graph_name_map` |
| `training_ready: false` | Train/train and test/train GED coverage |
| Runtime missing | `make setup`, Python paths, and `pip check` |
| Repository/code missing | Expected folder under `Models&Datasets/` |
| Checkpoint required | Training log and expected dynamic path |
| Executed but inaccurate | Training budget, split, target, features, and held-out metrics |
| GED 0 but prediction below 100% | Model error; the reference must not replace the prediction |
| GraphSim unavailable | Missing or invalid validation-only calibration |
| SEGMN out of memory | Batch size and validated node/edge caps |
| HPO config ignored | Dataset fingerprint or search-space version changed |

## Final Checklist

- [ ] Source, version, and redistribution terms are recorded.
- [ ] Graph construction and feature rules are fixed.
- [ ] Graph IDs are stable and unique.
- [ ] Split groups match the real data-generating unit.
- [ ] Target provenance is classified correctly.
- [ ] GED coverage is broad enough for the intended experiment.
- [ ] Manifest counts and graph previews were checked.
- [ ] Every required model has a dataset-specific checkpoint.
- [ ] Checkpoints record seed, split, target, and fingerprint metadata.
- [ ] Native and comparable outputs are both retained.
- [ ] Exact, approximate, unverified, and proxy results are separated.
- [ ] Final results include multiple seeds and failed runs.
- [ ] Source tests, GNN tests, preflight, and audits pass.

At that point the dataset is traceable from conversion to training and
evaluation. That is a stronger claim than simply seeing it in the dropdown.
