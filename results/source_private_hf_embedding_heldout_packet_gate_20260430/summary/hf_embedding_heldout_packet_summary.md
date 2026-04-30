# Frozen Embedding Held-Out Packet Summary

- pass gate: `False`
- rows: `60`
- pass rows: `20`
- near-miss rows: `32`
- semantic-anchor reference pass rows: `18/18`
- semantic-anchor min passing accuracy: `0.75`
- semantic-anchor min CI95 low: `0.45703125`

## Runs

| Run | Model | Mode | Pass rows | Direction pass | Max acc | Max delta | Max CI95 low | Min oracle |
|---|---|---|---:|---|---:|---:|---:|---:|
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_minilm_midlast_top20` | sentence-transformers/all-MiniLM-L6-v2 | `hf_mid_last_mean` | 4 | `{'core_to_holdout': False, 'holdout_to_core': True, 'same_family_all': True}` | 0.625 | 0.375 | 0.289 | 0.625 |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_minilm_last_top12` | sentence-transformers/all-MiniLM-L6-v2 | `hf_last_mean` | 4 | `{'core_to_holdout': False, 'holdout_to_core': True, 'same_family_all': True}` | 0.500 | 0.250 | 0.195 | 0.750 |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_minilm_top20_ridge005` | sentence-transformers/all-MiniLM-L6-v2 | `hf_last_mean` | 4 | `{'core_to_holdout': False, 'holdout_to_core': True, 'same_family_all': True}` | 0.500 | 0.250 | 0.195 | 0.750 |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_minilm_top20_ridge025_dec050` | sentence-transformers/all-MiniLM-L6-v2 | `hf_last_mean` | 3 | `{'core_to_holdout': False, 'holdout_to_core': True, 'same_family_all': True}` | 0.750 | 0.500 | 0.406 | 0.750 |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_bge_last` | BAAI/bge-small-en | `hf_last_mean` | 2 | `{'core_to_holdout': False, 'holdout_to_core': True, 'same_family_all': True}` | 0.500 | 0.250 | 0.199 | 0.625 |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_bge_last_top12` | BAAI/bge-small-en | `hf_last_mean` | 2 | `{'core_to_holdout': False, 'holdout_to_core': True, 'same_family_all': True}` | 0.500 | 0.250 | 0.199 | 0.625 |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_bge_midlast_top12` | BAAI/bge-small-en | `hf_mid_last_mean` | 1 | `{'core_to_holdout': False, 'holdout_to_core': True, 'same_family_all': False}` | 0.500 | 0.250 | 0.199 | 0.500 |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_minilm_hashed_last` | sentence-transformers/all-MiniLM-L6-v2 | `hashed_hf_last_mean` | 0 | `{'core_to_holdout': False, 'holdout_to_core': False, 'same_family_all': False}` | 0.625 | 0.375 | 0.289 | 0.625 |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_minilm_hashed_midlast` | sentence-transformers/all-MiniLM-L6-v2 | `hashed_hf_mid_last_mean` | 0 | `{'core_to_holdout': False, 'holdout_to_core': False, 'same_family_all': False}` | 0.625 | 0.375 | 0.289 | 0.625 |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_bge_hashed_last` | BAAI/bge-small-en | `hashed_hf_last_mean` | 0 | `{'core_to_holdout': False, 'holdout_to_core': False, 'same_family_all': False}` | 0.625 | 0.375 | 0.316 | 0.375 |

## Best Rows

| Run | Direction | Budget | Pass | Acc | Target | Best ctrl | Delta | CI95 low | Oracle | Top knock | Random knock | Controls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_minilm_top20_ridge025_dec050` | holdout_to_core | 4 | `False` | 0.750 | 0.250 | 0.250 | 0.500 | 0.406 | 0.875 | 1.000 | 0.906 | `True` |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_minilm_top20_ridge025_dec050` | same_family_all | 4 | `True` | 0.688 | 0.250 | 0.250 | 0.438 | 0.363 | 0.812 | 1.000 | 0.670 | `True` |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_minilm_top20_ridge025_dec050` | core_to_holdout | 4 | `False` | 0.625 | 0.250 | 0.250 | 0.375 | 0.316 | 0.750 | 1.000 | 0.542 | `True` |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_bge_hashed_last` | core_to_holdout | 8 | `False` | 0.625 | 0.250 | 0.250 | 0.375 | 0.316 | 0.750 | 1.000 | 0.365 | `True` |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_minilm_midlast_top20` | holdout_to_core | 4 | `True` | 0.625 | 0.250 | 0.250 | 0.375 | 0.289 | 1.000 | 1.000 | 0.500 | `True` |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_minilm_hashed_midlast` | holdout_to_core | 4 | `False` | 0.625 | 0.250 | 0.250 | 0.375 | 0.289 | 0.750 | 1.000 | 0.677 | `True` |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_minilm_midlast_top20` | holdout_to_core | 8 | `True` | 0.625 | 0.250 | 0.254 | 0.375 | 0.281 | 1.000 | 1.000 | 0.542 | `True` |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_minilm_top20_ridge025_dec050` | holdout_to_core | 8 | `True` | 0.625 | 0.250 | 0.266 | 0.375 | 0.281 | 1.000 | 1.000 | 0.208 | `True` |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_bge_hashed_last` | same_family_all | 4 | `False` | 0.562 | 0.250 | 0.250 | 0.312 | 0.254 | 0.562 | 1.000 | 0.800 | `True` |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_minilm_midlast_top20` | same_family_all | 4 | `True` | 0.562 | 0.250 | 0.250 | 0.312 | 0.242 | 0.812 | 1.000 | 0.525 | `True` |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_minilm_hashed_last` | same_family_all | 4 | `False` | 0.562 | 0.250 | 0.250 | 0.312 | 0.242 | 0.625 | 1.000 | 0.425 | `True` |
| `results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_minilm_hashed_midlast` | same_family_all | 4 | `False` | 0.562 | 0.250 | 0.250 | 0.312 | 0.242 | 0.688 | 1.000 | 0.550 | `True` |

## Interpretation

Frozen embedding receivers recover part of the semantic-anchor held-out packet signal with clean source-destroying controls, but none of the tested BGE/MiniLM variants clears the strict bidirectional gate. This weakens the hypothesis that a generic frozen text embedding alone can replace the explicit public semantic-anchor lexicon; the live next branch is a learned receiver or a better public ontology calibration layer, not another fixed-basis packet tweak.
