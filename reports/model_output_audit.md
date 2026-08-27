# Model Output Audit

Technical integrity verifies execution, input binding, bounds, and GED-scale conversion. It does not certify checkpoint accuracy.

- Dataset: `aids700nef`
- Pair: `AIDS700nef/train/4.gexf` vs `AIDS700nef/test/6.gexf`
- GED reference: `12.0`
- Reference kind: `exact`
- Technical integrity: `5/5`

| Model | Integrity | Native | Comparable | Pred. GED | GED error | Symmetry gap | Identity similarity |
|---|---|---:|---:|---:|---:|---:|---:|
| SimGNN | pass | 0.276024 | 0.276024 | 12.872686 | 0.872686 | 0.043458 | 0.930807 |
| Multi-Scale Convolutional Set Matching | pass | 0.617309 | 0.502016 | 6.891225 | 5.108775 | 0.104805 | 0.404063 |
| SEGMN | pass | 0.450734 | 0.450734 | 7.968769 | 4.031231 | 0.011613 | 0.997728 |
| Graph Fusion | pass | 0.310342 | 0.310342 | 11.700809 | 0.299191 | 0.000314 | 0.972750 |
| Graph2Region | pass | 0.268481 | 0.268481 | 13.149743 | 1.149743 | 0.000000 | 1.000000 |

Identity and symmetry columns are checkpoint-behavior diagnostics, not technical execution criteria. Weak values indicate model-fit limitations.
