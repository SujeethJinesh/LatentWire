# Toy Feature Atom Stack Bridge

- Seed: `17`
- Train examples: `72`
- Test examples: `48`
- Shared features: `6`
- Route families: `4`
- Atoms per family: `4`
- Protected shared rows: `2`
- Protected atoms: `3`

| Method | Test Acc | Test MSE | Shared rec | Atom rec | Shared H | Atom H | Bytes proxy | Compute proxy | Help vs raw | Harm vs raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_ridge | 0.6458 | 1.1905 | 0.0000 | 0.0000 | 1.6335 | 2.2103 | 1024.0000 | 18432.0000 | 0.0000 | 0.0000 |
| shared_feature_only | 0.4167 | 5.5088 | 0.2489 | 0.0000 | 1.2681 | 1.2255 | 912.0000 | 41472.0000 | 0.0000 | 0.2292 |
| route_atom_only | 0.5833 | 2.0779 | 0.0000 | 0.5614 | 1.5512 | 2.1950 | 3072.0000 | 110592.0000 | 0.0000 | 0.0625 |
| stacked_feature_atom | 0.8542 | 1.7413 | 0.2559 | 0.3948 | 1.6853 | 1.9492 | 5920.0000 | 190080.0000 | 0.2083 | 0.0000 |
| protected_stacked_feature_atom | 0.8542 | 1.7427 | 0.2559 | 0.3948 | 1.6853 | 1.9492 | 6080.0000 | 209088.0000 | 0.2083 | 0.0000 |
| oracle | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.7383 | 2.3603 | 0.0000 | 0.0000 | 0.3542 | 0.0000 |
