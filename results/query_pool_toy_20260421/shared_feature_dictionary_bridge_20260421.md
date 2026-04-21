# Toy Shared Feature Dictionary Bridge

- seed: `4`
- dim: `12`
- shared features: `4`
- source private features: `3`
- target private features: `3`

| Method | Test Acc | Test MSE | Shared recovery | Sparsity | Align residual | Bytes proxy | Compute proxy | Acc delta | MSE delta | Help vs raw | Harm vs raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_residual_bridge | 0.3646 | 1.1263 | 0.2569 | 0.4084 | - | 624.0000 | 13824.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| separate_per_model_dictionaries | 0.4167 | 0.2763 | 0.3125 | 0.4141 | 0.5915 | 1360.0000 | 184320.0000 | 0.0521 | -0.8500 | 0.0521 | 0.0000 |
| shared_dictionary_crosscoder | 0.5417 | 0.2642 | 0.3125 | 0.4324 | 0.7390 | 1360.0000 | 184320.0000 | 0.1771 | -0.8621 | 0.1771 | 0.0000 |
| symmetry_aware_shared_dictionary | 0.4688 | 0.3026 | 0.3056 | 0.4163 | 0.6073 | 1360.0000 | 230400.0000 | 0.1042 | -0.8237 | 0.1042 | 0.0000 |
| oracle_upper_bound | 0.5938 | 0.0000 | 0.6597 | 0.8695 | 0.0000 | 0.0000 | 0.0000 | 0.2292 | -1.1263 | 0.2292 | 0.0000 |
