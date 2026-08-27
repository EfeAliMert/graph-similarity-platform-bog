# Model Output Audit

Technical integrity verifies execution, input binding, bounds, and GED-scale conversion. It does not certify checkpoint accuracy.

- Dataset: `enzymes`
- Pair: `ENZYMES/train/0.gexf` vs `ENZYMES/test/480.gexf`
- GED reference: `None`
- Reference kind: `None`
- Technical integrity: `5/5`

| Model | Integrity | Native | Comparable | Pred. GED | GED error | Symmetry gap | Identity similarity |
|---|---|---:|---:|---:|---:|---:|---:|
| SimGNN | pass | 0.011535 | 0.011535 | 109.328562 | - | 0.001047 | 0.716002 |
| Multi-Scale Convolutional Set Matching | pass | 0.048755 | 0.013358 | 105.733313 | - | 0.000765 | 0.728149 |
| SEGMN | pass | 0.485676 | 0.485676 | 17.694241 | - | 0.000021 | 0.485576 |
| Graph Fusion | pass | 0.476358 | 0.476358 | 18.168848 | - | 0.016836 | 0.999186 |
| Graph2Region | pass | 0.232612 | 0.232612 | 35.730382 | - | 0.005274 | 0.242629 |

Identity and symmetry columns are checkpoint-behavior diagnostics, not technical execution criteria. Weak values indicate model-fit limitations.
