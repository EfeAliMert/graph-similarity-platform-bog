# Dataset Target Audit

GED is treated as symmetric. Directional records are collapsed by unordered graph id and the minimum finite non-negative value is used. For approximate solvers this is the tighter valid upper bound.

| Dataset | Graphs | Reference | Raw conflicts | Coverage | Canonical conflicts | Status |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| AIDS700nef | 700 | exact | 0 | 1.000 | 0 | exact_verified |
| LINUX | 1000 | exact | 0 | 1.000 | 0 | exact_verified |
| IMDBMulti | 1500 | approximate_benchmark | 187880 | 1.000 | 0 | approximate_verified |
| PTC | 344 | approximate_benchmark | 33283 | 1.000 | 0 | approximate_verified |
| MUTAG | 188 | structural_proxy | n/a | n/a | n/a | proxy_only |
| PROTEINS | 1113 | structural_proxy | n/a | n/a | n/a | proxy_only |
| ENZYMES | 600 | structural_proxy | n/a | n/a | n/a | proxy_only |

## Interpretation

- **AIDS700nef:** Valid exact-GED reference.
- **LINUX:** Valid exact-GED reference.
- **IMDBMulti:** Valid published approximate benchmark reference; values are upper bounds and must not be reported as exact GED.
- **PTC:** Valid published approximate benchmark reference; values are upper bounds and must not be reported as exact GED.
- **MUTAG:** The checkpoint can be evaluated only for fidelity to the declared structural proxy; no exact or approximate GED benchmark claim is valid.
- **PROTEINS:** The checkpoint can be evaluated only for fidelity to the declared structural proxy; no exact or approximate GED benchmark claim is valid.
- **ENZYMES:** The checkpoint can be evaluated only for fidelity to the declared structural proxy; no exact or approximate GED benchmark claim is valid.
