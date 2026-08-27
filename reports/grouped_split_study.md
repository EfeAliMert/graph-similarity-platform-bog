# Grouped Split Study

Pair-disjoint validation keeps `(A,B)` and `(B,A)` on the same side of the split. It can still share individual graphs or subjects. Subject-disjoint splits graph identities first, then builds pairs.

## Synthetic subjects

Twelve graphs from six subjects. Pair-disjoint validation can still share a subject/graph identity; subject-disjoint splits graphs first.

| Strategy | Train graphs | Val graphs | Graph overlap | Pair overlap |
|---|---:|---:|---:|---:|
| pair-disjoint | 12 | 11 | 11 | 0 |
| subject-disjoint | 9 | 3 | 0 | 0 |
