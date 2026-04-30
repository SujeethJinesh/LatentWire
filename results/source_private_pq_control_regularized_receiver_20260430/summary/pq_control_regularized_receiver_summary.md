# PQ Control-Regularized Receiver Summary

- pass gate: `True`
- rows: `6`
- overlap pass rows: `3/4`
- disjoint pass rows: `0/2`
- min low-control overlap learned accuracy: `0.504`
- max low-control overlap best control accuracy: `0.298`
- max disjoint L2 accuracy: `0.270`

| Run | Disjoint | Remap | Pass | Learned | L2 | Target | Best control | Deranged | CI low vs control |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `results/source_private_pq_control_regularized_receiver_20260430/n256_remap101_utility_protected_hadamard` | `True` | 101 | `False` | 0.250 | 0.270 | 0.250 | label_shuffled_ridge 0.250 | 0.250 | 0.000 |
| `results/source_private_pq_control_regularized_receiver_20260430/n500_overlap_remap101_utility_protected_hadamard` | `False` | 101 | `False` | 0.250 | 0.504 | 0.250 | label_shuffled_ridge 0.250 | 0.250 | 0.000 |
| `results/source_private_pq_control_regularized_receiver_20260430/n500_overlap_remap101_utility_protected_hadamard_lowcontrol` | `False` | 101 | `True` | 0.504 | 0.504 | 0.250 | answer_masked_source 0.268 | 0.180 | 0.178 |
| `results/source_private_pq_control_regularized_receiver_20260430/n500_overlap_remap103_utility_protected_hadamard_lowcontrol` | `False` | 103 | `True` | 0.504 | 0.504 | 0.250 | random_same_byte 0.298 | 0.158 | 0.142 |
| `results/source_private_pq_control_regularized_receiver_20260430/n500_overlap_remap107_utility_protected_hadamard_lowcontrol` | `False` | 107 | `True` | 0.516 | 0.516 | 0.250 | permuted_codes 0.262 | 0.124 | 0.200 |
| `results/source_private_pq_control_regularized_receiver_20260430/n500_disjoint_remap101_utility_protected_hadamard_lowcontrol` | `True` | 101 | `False` | 0.264 | 0.264 | 0.250 | permuted_codes 0.288 | 0.234 | -0.080 |

The low-control learned PQ receiver preserves deterministic PQ on the established exact-ID overlap surface across remaps, while disjoint train/eval IDs collapse the underlying PQ signal. This is a bounded positive diagnostic for learned reception and a stronger blocker against using PQ as an ICLR headline until the packet source is disjoint-safe.
