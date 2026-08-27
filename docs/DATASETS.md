# Dataset Sources and Targets

Install every registered collection with:

```bash
make setup
make datasets
```

The source repository does not contain third-party dataset bytes. The installer
downloads the four GED benchmark collections directly from the GraphSim
authors' [shared benchmark folder](https://drive.google.com/drive/folders/1JcAgWKYC41687UeiLaFg-QlPmIpZvWhT?usp=sharing).
Each downloaded archive and GED/MCS map must match the byte count and SHA-256
in `configs/dataset_sources.json` before it is installed. This check is
especially important because the target maps use Python pickle files.

| Dataset | Graphs | Pair target used by the platform |
| --- | ---: | --- |
| AIDS700nef | 700 | Exact A* GED and MCS benchmark map |
| LINUX | 1,000 | Exact A* GED and MCS benchmark map |
| IMDBMulti | 1,500 | Approximate GED upper bound and MCS benchmark map |
| PTC | 344 | Approximate GED upper bound and MCS benchmark map |

TUDataset dumps (MUTAG, PROTEINS, ENZYMES) were exported with
`scripts/download_real_graph_datasets.py` into the same GEXF train/test layout
as the GED sets. `make datasets` runs this export through PyTorch Geometric.

| Dataset | Graphs | Local archive | Pair target available locally |
| --- | ---: | --- | --- |
| MUTAG | 188 | `Models&Datasets/drive-download-20260630T100606Z-3-001/MUTAG.zip` | Structural proxy only |
| PROTEINS | 1,113 | `Models&Datasets/drive-download-20260630T100606Z-3-001/PROTEINS.zip` | Structural proxy only |
| ENZYMES | 600 | `Models&Datasets/drive-download-20260630T100606Z-3-001/ENZYMES.zip` | Structural proxy only |

The archives contain `train/*.gexf` and `test/*.gexf`, so they can be selected
and previewed like the GED datasets. Their original labels are graph-class
labels, not pairwise edit distances. The platform therefore does not present
their structural proxy as exact GED.

The root MIT license covers the platform code only. Dataset-specific terms
still apply to every downloaded local copy; see `docs/THIRD_PARTY.md`.
