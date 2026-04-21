# Toy Mixed-Bit Route-Atom Allocator

- EXL2/AWQ-style allocator toy: every route atom remains active, and only per-atom precision changes.
- Mixed allocators target the configured average bpw by assigning high bits to selected route atoms and low bits elsewhere.
- Patch-rank correlation measures agreement with calibration exact single-atom high-bit gains.

| Method | Accuracy | Acc delta | MSE | MSE delta | Achieved bpw | Bit histogram | Patch-rank corr | Outlier protection | Exact overlap | Feature overlap | Stability | Bytes proxy | Compute proxy | Help vs 3-bit | Harm vs 3-bit |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform_3_bit | 0.2250 | 0.0000 | 0.1269 | 0.0000 | 3.0000 | `{"3": 32}` | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 244.0000 | 640.0000 | 0.0000 | 0.0000 |
| uniform_4_bit | 1.0000 | 0.7750 | 0.0303 | -0.0966 | 4.0000 | `{"4": 32}` | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 324.0000 | 640.0000 | 0.7750 | 0.0000 |
| quant_error_target_bpw_allocator | 1.0000 | 0.7750 | 0.0314 | -0.0955 | 3.9375 | `{"3": 26, "8": 6}` | 0.8886 | 0.6000 | 0.6667 | 0.1667 | 1.0000 | 319.0000 | 676.0000 | 0.7750 | 0.0000 |
| exact_patch_target_bpw_allocator | 1.0000 | 0.7750 | 0.0388 | -0.0881 | 3.9375 | `{"3": 26, "8": 6}` | 1.0000 | 0.6000 | 1.0000 | 0.1667 | 1.0000 | 319.0000 | 676.0000 | 0.7750 | 0.0000 |
| universal_feature_persistence_allocator | 0.7438 | 0.5188 | 0.0334 | -0.0935 | 3.9375 | `{"3": 26, "8": 6}` | 0.8402 | 0.6000 | 0.6667 | 0.1667 | 1.0000 | 319.0000 | 676.0000 | 0.5188 | 0.0000 |
| random_allocator | 0.2250 | 0.0000 | 0.0924 | -0.0344 | 3.9375 | `{"3": 26, "8": 6}` | 0.0418 | 0.2000 | 0.1667 | 0.0000 | 0.0909 | 319.0000 | 676.0000 | 0.0000 | 0.0000 |
| oracle_allocator | 1.0000 | 0.7750 | 0.0388 | -0.0881 | 3.9375 | `{"3": 26, "8": 6}` | 0.9989 | 0.6000 | 1.0000 | 0.1667 | 1.0000 | 319.0000 | 676.0000 | 0.7750 | 0.0000 |
