# ARC-Challenge Behavior-Atom Decoder Packet Gate

- date: `2026-05-04`
- pass gate: `False`
- train/test disagreement rows: `16` / `16`
- matched accuracy: `0.250000`
- target-only accuracy: `0.250000`
- best required control: `source_row_shuffle`
- best required control accuracy: `0.437500`
- worst required paired CI95 low: `-0.437500`
- fired rows: `10`
- helps/harms vs target: `2` / `2`
- packet bytes/row: `3.500`
- same-byte visible-text budget: `4`

## Strict Controls

| Control | Accuracy | Delta | CI95 low |
|---|---:|---:|---:|
| `target_only` | 0.250000 | 0.000000 | -0.250000 |
| `target_decoder_only` | 0.187500 | 0.062500 | -0.125000 |
| `target_derived_packet` | 0.187500 | 0.062500 | 0.000000 |
| `zero_source` | 0.312500 | -0.062500 | -0.250000 |
| `source_row_shuffle` | 0.437500 | -0.187500 | -0.437500 |
| `same_source_choice_row_shuffle` | 0.187500 | 0.062500 | -0.125000 |
| `atom_shuffle` | 0.187500 | 0.062500 | -0.125000 |
| `coefficient_shuffle` | 0.187500 | 0.062500 | -0.125000 |
| `top_atom_knockout` | 0.312500 | -0.062500 | -0.187500 |
| `candidate_roll` | 0.187500 | 0.062500 | 0.000000 |
| `candidate_derangement` | 0.250000 | 0.000000 | -0.187500 |
| `packet_only_source_index` | 0.312500 | -0.062500 | -0.437500 |
| `source_rank_control` | 0.312500 | -0.062500 | -0.437500 |
| `source_score_control` | 0.312500 | -0.062500 | -0.437500 |
| `source_score_quantized_control` | 0.312500 | -0.062500 | -0.375000 |
| `same_byte_visible_text` | 0.312500 | -0.062500 | -0.312500 |
| `qwen_substituted_packet` | 0.437500 | -0.187500 | -0.437500 |

## Interpretation

This gate tests whether source-hidden innovations become more useful when the sparse atom basis is trained toward target residual behavior rather than unsupervised PCA variance. It passes only if the matched packet beats target-only, target-decoder-only, target-derived packets, same-source-choice and generic wrong-row packets, atom/coefficient destruction, candidate roll/derangement, source-index/rank/score, same-byte text, and Qwen-substitution controls with positive paired uncertainty.
