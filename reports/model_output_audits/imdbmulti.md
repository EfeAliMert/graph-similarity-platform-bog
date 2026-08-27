# Model Output Audit

Technical integrity verifies execution, input binding, bounds, and GED-scale conversion. It does not certify checkpoint accuracy.

- Dataset: `imdbmulti`
- Pair: `IMDBMulti/train/0.gexf` vs `IMDBMulti/test/8.gexf`
- GED reference: `27.0`
- Reference kind: `approximate_benchmark`
- Technical integrity: `5/5`

| Model | Integrity | Native | Comparable | Pred. GED | GED error | Symmetry gap | Identity similarity |
|---|---|---:|---:|---:|---:|---:|---:|
| SimGNN | pass | 0.349300 | 0.349300 | 10.518241 | 16.481759 | 0.011933 | 0.571362 |
| Multi-Scale Convolutional Set Matching | pass | 0.177182 | 0.084394 | 24.722567 | 2.277433 | 0.074215 | 0.927749 |
| SEGMN | pass | 0.193660 | 0.193660 | 16.416536 | 10.583464 | 0.104844 | 0.999565 |
| Graph Fusion | pass | 0.524965 | 0.524965 | 6.444246 | 20.555754 | 0.077940 | 0.953636 |
| Graph2Region | pass | 0.281213 | 0.281213 | 12.686428 | 14.313572 | 0.064902 | 0.990277 |

Identity and symmetry columns are checkpoint-behavior diagnostics, not technical execution criteria. Weak values indicate model-fit limitations.
