# Third-Party Inventory

These repos and datasets stay out of the source Git tree. MIT applies to the
platform code here, not to the rows below.

| Component | Local path | Upstream | Local license evidence | Use in platform |
|---|---|---|---|---|
| SimGNN community implementation | `Models&Datasets/SimGNN-v_00001` | <https://github.com/benedekrozemberczki/SimGNN> | Local `LICENSE` file | SimGNN architecture and training entrypoint |
| GraphSim | `Models&Datasets/GraphSim-master` | <https://github.com/yunshengb/GraphSim> | No license file found in local archive | Authors' GraphSim architecture through a TensorFlow compatibility layer |
| SEGMN | `Models&Datasets/SEGMN-main` | <https://github.com/tourist-wwj/SEGMN> | No license file found in the pinned upstream checkout | Authors' public SEGMN architecture |
| Graph Fusion Model | `Models&Datasets/GFM-code` | <https://github.com/LLiRarry/GFM-code> | No license file found in local clone | Paper-linked GMS architecture |
| Graph2Region | `Models&Datasets/Graph2Region-main` | <https://github.com/liuzhouyang/Graph2Region> | Local `LICENSE` file | Authors' G2R architecture |
| GED benchmark archives | `Models&Datasets/drive-download-20260630T100606Z-3-001` | [GraphSim authors' shared folder](https://drive.google.com/drive/folders/1JcAgWKYC41687UeiLaFg-QlPmIpZvWhT?usp=sharing) | Dataset-specific terms; not covered by root license | Graph files and GED/MCS maps, installed with checksum verification |
| TUDataset downloads | Registered local dataset archives | <https://chrsmrrs.github.io/datasets/> | Dataset-specific terms | MUTAG, PROTEINS, ENZYMES |

If a local copy has no license file, it is not covered by this repo's MIT
license. Check upstream terms before publishing model code, data, or weights
together with this project.

`make datasets` transfers data directly from the listed upstream services. It
does not grant redistribution rights or place those files under the platform's
license.

`make models` clones the upstream repositories directly. The upstream source
trees remain under their authors' terms; this repository stores only its
installer, pinned identities, and narrow compatibility patches.
