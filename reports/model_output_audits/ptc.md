# Model Output Audit

Technical integrity verifies execution, input binding, bounds, and GED-scale conversion. It does not certify checkpoint accuracy.

- Dataset: `ptc`
- Pair: `PTC/train/1.gexf` vs `PTC/test/0.gexf`
- GED reference: `40.0`
- Reference kind: `approximate_benchmark`
- Technical integrity: `5/5`

| Model | Integrity | Native | Comparable | Pred. GED | GED error | Symmetry gap | Identity similarity |
|---|---|---:|---:|---:|---:|---:|---:|
| SimGNN | pass | 0.172875 | 0.172875 | 22.817411 | 17.182589 | 0.059830 | 0.967624 |
| Multi-Scale Convolutional Set Matching | pass | 0.147596 | 0.065007 | 35.532324 | 4.467676 | 0.002259 | 0.407759 |
| SEGMN | pass | 0.062836 | 0.062836 | 35.973862 | 4.026138 | 0.004367 | 0.616460 |
| Graph Fusion | pass | 0.053123 | 0.053123 | 38.156906 | 1.843094 | 0.000979 | 0.384474 |
| Graph2Region | pass | 0.121862 | 0.121862 | 27.363253 | 12.636747 | 0.027134 | 0.767187 |

Identity and symmetry columns are checkpoint-behavior diagnostics, not technical execution criteria. Weak values indicate model-fit limitations.
