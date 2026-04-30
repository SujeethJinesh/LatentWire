# Source-Private Masked Consistency Receiver Summary

- pass gate: `True`
- runs: `3`
- n256 runs: `2`
- min n256 learned matched accuracy: `0.957`
- min n256 lift vs target: `0.707`
- min n256 lift vs best control: `0.676`
- min n256 CI95 low vs target: `0.652`
- min n256 CI95 low vs best control: `0.617`
- n256 learned-minus-Hamming range: `-0.020` to `0.016`

| Run | n | pass | learned | Hamming | target | best control | lift vs control | CI low vs control | learned-Hamming |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `results/source_private_masked_consistency_receiver_smoke_20260430/n64_seed29_30_budget6` | 64 | `True` | 0.969 | 0.969 | 0.250 | `shuffled_source` 0.281 | 0.688 | 0.562 | 0.000 |
| `results/source_private_masked_consistency_receiver_smoke_20260430/n256_seed29_30_budget6` | 256 | `True` | 0.977 | 0.961 | 0.250 | `wrong_projection_source` 0.258 | 0.719 | 0.664 | 0.016 |
| `results/source_private_masked_consistency_receiver_smoke_20260430/n256_seed31_32_budget6` | 256 | `True` | 0.957 | 0.977 | 0.250 | `wrong_projection_source` 0.281 | 0.676 | 0.617 | -0.020 |

A one-step learned masked-consistency receiver over 6-byte learned syndrome packets preserves most deterministic packet utility while suppressing destructive-control leakage. It is not yet a fully table-free semantic receiver because it uses public candidate/code features and is compared against deterministic Hamming decoding.
