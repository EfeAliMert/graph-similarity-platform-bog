# Model Output Audit

Technical integrity verifies execution, input binding, bounds, and GED-scale conversion. It does not certify checkpoint accuracy.

- Dataset: `mutag`
- Pair: `MUTAG/train/0.gexf` vs `MUTAG/test/150.gexf`
- GED reference: `None`
- Reference kind: `None`
- Technical integrity: `5/5`

| Model | Integrity | Native | Comparable | Pred. GED | GED error | Symmetry gap | Identity similarity |
|---|---|---:|---:|---:|---:|---:|---:|
| SimGNN | pass | 0.269636 | 0.269636 | 19.004888 | - | 0.017128 | 0.818255 |
| Multi-Scale Convolutional Set Matching | pass | 0.432440 | 0.301921 | 17.365044 | - | 0.005271 | 0.953134 |
| SEGMN | pass | 0.495421 | 0.495421 | 10.184030 | - | 0.000115 | 0.495378 |
| Graph Fusion | pass | 0.424664 | 0.424664 | 12.418624 | - | 0.011590 | 0.967007 |
| Graph2Region | pass | 0.503298 | 0.503298 | 9.955310 | - | 0.014426 | 0.519375 |

Identity and symmetry columns are checkpoint-behavior diagnostics, not technical execution criteria. Weak values indicate model-fit limitations.
