# Model Output Audit

Technical integrity verifies execution, input binding, bounds, and GED-scale conversion. It does not certify checkpoint accuracy.

- Dataset: `proteins`
- Pair: `PROTEINS/train/0.gexf` vs `PROTEINS/test/890.gexf`
- GED reference: `None`
- Reference kind: `None`
- Technical integrity: `5/5`

| Model | Integrity | Native | Comparable | Pred. GED | GED error | Symmetry gap | Identity similarity |
|---|---|---:|---:|---:|---:|---:|---:|
| SimGNN | pass | 0.000830 | 0.000830 | 166.722057 | - | 0.000686 | 0.675601 |
| Multi-Scale Convolutional Set Matching | pass | 0.015667 | 0.002639 | 139.528487 | - | 0.000267 | 0.770756 |
| SEGMN | pass | 0.462792 | 0.462792 | 18.106230 | - | 0.000145 | 0.461352 |
| Graph Fusion | pass | 0.248018 | 0.248018 | 32.764923 | - | 0.027225 | 0.999084 |
| Graph2Region | pass | 0.146263 | 0.146263 | 45.175139 | - | 0.019494 | 0.163389 |

Identity and symmetry columns are checkpoint-behavior diagnostics, not technical execution criteria. Weak values indicate model-fit limitations.
