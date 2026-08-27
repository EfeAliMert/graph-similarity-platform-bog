# Model Output Audit

Technical integrity verifies execution, input binding, bounds, and GED-scale conversion. It does not certify checkpoint accuracy.

- Dataset: `linux`
- Pair: `LINUX/train/3.gexf` vs `LINUX/test/382.gexf`
- GED reference: `10.0`
- Reference kind: `exact`
- Technical integrity: `5/5`

| Model | Integrity | Native | Comparable | Pred. GED | GED error | Symmetry gap | Identity similarity |
|---|---|---:|---:|---:|---:|---:|---:|
| SimGNN | pass | 0.280973 | 0.280973 | 7.616975 | 2.383025 | 0.003878 | 0.866934 |
| Multi-Scale Convolutional Set Matching | pass | 0.351577 | 0.224624 | 8.959954 | 1.040046 | 0.000000 | 0.891118 |
| SEGMN | pass | 0.189304 | 0.189304 | 9.986410 | 0.013590 | 0.050474 | 0.996437 |
| Graph Fusion | pass | 0.169556 | 0.169556 | 10.647421 | 0.647421 | 0.003949 | 0.969551 |
| Graph2Region | pass | 0.327746 | 0.327746 | 6.693092 | 3.306908 | 0.104440 | 0.960084 |

Identity and symmetry columns are checkpoint-behavior diagnostics, not technical execution criteria. Weak values indicate model-fit limitations.
