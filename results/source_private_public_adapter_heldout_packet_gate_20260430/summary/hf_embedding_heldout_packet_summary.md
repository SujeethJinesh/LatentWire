# Public Adapter Held-Out Packet Summary

- pass gate: `False`
- rows: `36`
- pass rows: `3`
- near-miss rows: `3`
- semantic-anchor reference pass rows: `18/18`
- semantic-anchor min passing accuracy: `0.75`
- semantic-anchor min CI95 low: `0.45703125`

## Runs

| Run | Model | Mode | Adapter target | Pass rows | Direction pass | Max acc | Max delta | Max CI95 low | Min oracle |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| `results/source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher_permuted` | sentence-transformers/all-MiniLM-L6-v2 | `hf_last_mean` | `permuted_semantic_anchor_teacher` | 2 | `{'core_to_holdout': True, 'holdout_to_core': False, 'same_family_all': True}` | 0.625 | 0.375 | 0.316 | 0.625 |
| `results/source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher_top8_dec040` | sentence-transformers/all-MiniLM-L6-v2 | `hf_last_mean` | `semantic_anchor_teacher` | 1 | `{'core_to_holdout': False, 'holdout_to_core': True, 'same_family_all': False}` | 0.875 | 0.625 | 0.566 | 0.625 |
| `results/source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher` | sentence-transformers/all-MiniLM-L6-v2 | `hf_last_mean` | `semantic_anchor_teacher` | 0 | `{'core_to_holdout': False, 'holdout_to_core': False, 'same_family_all': False}` | 0.875 | 0.625 | 0.566 | 0.625 |
| `results/source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher_trainonly_top8_dec040` | sentence-transformers/all-MiniLM-L6-v2 | `hf_last_mean` | `semantic_anchor_teacher` | 0 | `{'core_to_holdout': False, 'holdout_to_core': False, 'same_family_all': False}` | 0.875 | 0.625 | 0.562 | 0.625 |

## Best Rows

| Run | Direction | Budget | Pass | Acc | Target | Best ctrl | Delta | CI95 low | Oracle | Top knock | Random knock | Controls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `results/source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher_top8_dec040` | holdout_to_core | 4 | `True` | 0.875 | 0.250 | 0.250 | 0.625 | 0.566 | 1.000 | 1.000 | 0.500 | `True` |
| `results/source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher` | same_family_all | 4 | `False` | 0.875 | 0.250 | 0.312 | 0.625 | 0.562 | 0.875 | 1.000 | 0.194 | `False` |
| `results/source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher_top8_dec040` | same_family_all | 4 | `False` | 0.875 | 0.250 | 0.312 | 0.625 | 0.562 | 0.875 | 1.000 | 0.463 | `False` |
| `results/source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher_trainonly_top8_dec040` | same_family_all | 4 | `False` | 0.875 | 0.250 | 0.312 | 0.625 | 0.562 | 0.875 | 1.000 | 0.463 | `False` |
| `results/source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher` | core_to_holdout | 4 | `False` | 0.875 | 0.250 | 0.375 | 0.625 | 0.566 | 0.750 | 1.000 | 0.194 | `False` |
| `results/source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher` | core_to_holdout | 8 | `False` | 0.875 | 0.250 | 0.375 | 0.625 | 0.566 | 0.875 | 1.000 | 0.100 | `False` |
| `results/source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher` | holdout_to_core | 4 | `False` | 0.875 | 0.250 | 0.375 | 0.625 | 0.566 | 1.000 | 1.000 | 0.175 | `False` |
| `results/source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher_top8_dec040` | core_to_holdout | 4 | `False` | 0.875 | 0.250 | 0.375 | 0.625 | 0.566 | 0.750 | 1.000 | 0.444 | `False` |
| `results/source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher_top8_dec040` | core_to_holdout | 8 | `False` | 0.875 | 0.250 | 0.375 | 0.625 | 0.566 | 0.875 | 1.000 | 0.100 | `False` |
| `results/source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher` | core_to_holdout | 2 | `False` | 0.750 | 0.250 | 0.250 | 0.500 | 0.434 | 0.625 | 1.000 | 1.000 | `True` |
| `results/source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher` | same_family_all | 2 | `False` | 0.750 | 0.250 | 0.254 | 0.500 | 0.441 | 0.625 | 1.000 | 1.000 | `True` |
| `results/source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher` | holdout_to_core | 2 | `False` | 0.750 | 0.250 | 0.254 | 0.500 | 0.438 | 0.625 | 1.000 | 1.000 | `True` |

## Interpretation

Public semantic-anchor teacher adapters can produce large matched-packet lifts, but this sweep does not clear the strict bidirectional gate. The permuted-teacher negative control also passes some individual rows, so the branch is alive only if the next receiver adds candidate-local normalization or residual/codebook conditioning that collapses shuffled, deranged, and permuted controls while preserving the matched source packet.
