# ARC-Challenge Residual/Syndrome Packet Gate

- date: `2026-05-04`
- pass gate: `False`
- train/test disagreement rows: `32` / `32`
- matched accuracy: `0.218750`
- target-only accuracy: `0.250000`
- best required control: `qwen_substituted_packet`
- best required control accuracy: `0.375000`
- worst required paired CI95 low: `-0.328906`
- fired rows: `14`
- helps/harms vs target: `3` / `4`
- packet bytes/row: `1.375`

## Strict Controls

| Control | Accuracy | Delta | CI95 low |
|---|---:|---:|---:|
| `target_only` | 0.250000 | -0.031250 | -0.187500 |
| `target_derived_packet` | 0.250000 | -0.031250 | -0.187500 |
| `zero_source` | 0.250000 | -0.031250 | -0.187500 |
| `source_row_shuffle` | 0.125000 | 0.093750 | -0.062500 |
| `header_shuffle` | 0.187500 | 0.031250 | -0.156250 |
| `syndrome_bit_shuffle` | 0.218750 | 0.000000 | -0.125000 |
| `parity_flip` | 0.250000 | -0.031250 | -0.187500 |
| `wrong_parity_matrix` | 0.312500 | -0.093750 | -0.187500 |
| `target_side_info_removed` | 0.218750 | 0.000000 | 0.000000 |
| `candidate_roll` | 0.250000 | -0.031250 | -0.250000 |
| `candidate_derangement` | 0.250000 | -0.031250 | -0.281250 |
| `packet_only_source_index` | 0.312500 | -0.093750 | -0.312500 |
| `source_rank_control` | 0.312500 | -0.093750 | -0.328906 |
| `source_score_control` | 0.312500 | -0.093750 | -0.312500 |
| `source_score_quantized_control` | 0.312500 | -0.093750 | -0.312500 |
| `same_byte_visible_text` | 0.312500 | -0.093750 | -0.312500 |
| `qwen_substituted_packet` | 0.375000 | -0.156250 | -0.312500 |

## Interpretation

This gate tests whether a fixed-byte pairwise residual/syndrome packet can transmit source innovation without exposing source text, KV, hidden states, or raw score vectors. It passes only if the decoded packet beats target-only, target-derived, wrong-row, syndrome-destroyed, source-choice/source-score, same-byte visible text, and Qwen-substitution controls with positive paired uncertainty.
