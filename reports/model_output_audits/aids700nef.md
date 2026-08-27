# Model Output Audit

Technical integrity verifies execution, input binding, bounds, and GED-scale conversion. It does not certify checkpoint accuracy.

- Dataset: `aids700nef`
- Pair: `AIDS700nef/train/4.gexf` vs `AIDS700nef/test/6.gexf`
- GED reference: `12.0`
- Reference kind: `exact`
- Technical integrity: `5/5`

| Model | Integrity | Native | Comparable | Pred. GED | GED error | Symmetry gap | Identity similarity |
|---|---|---:|---:|---:|---:|---:|---:|
| SimGNN | pass | 0.373568 | 0.373568 | 9.846542 | 2.153458 | 0.005280 | 0.995509 |
| Multi-Scale Convolutional Set Matching | pass | 0.593415 | 0.474490 | 7.455154 | 4.544846 | 0.004317 | 0.422198 |
| SEGMN | pass | 0.210638 | 0.210638 | 15.576130 | 3.576130 | 0.027089 | 0.999843 |
| Graph Fusion | pass | 0.305706 | 0.305706 | 11.851318 | 0.148682 | 0.007876 | 0.802884 |
| Graph2Region | pass | 0.264531 | 0.264531 | 13.297969 | 1.297969 | 0.039164 | 0.888745 |

Identity and symmetry columns are checkpoint-behavior diagnostics, not technical execution criteria. Weak values indicate model-fit limitations.
