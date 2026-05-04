# ARC-Challenge Behavior-Atom Decoder Packet Gate

- date: `2026-05-04`
- pass gate: `False`
- train/test disagreement rows: `16` / `16`
- matched accuracy: `0.375000`
- target-only accuracy: `0.250000`
- best required control: `top_atom_knockout`
- best required control accuracy: `0.500000`
- worst required paired CI95 low: `-0.375000`
- fired rows: `7`
- helps/harms vs target: `2` / `0`
- packet bytes/row: `7.000`
- same-byte visible-text budget: `7`

## Strict Controls

| Control | Accuracy | Delta | CI95 low |
|---|---:|---:|---:|
| `atom_shuffle` | 0.125000 | 0.250000 | 0.062500 |
| `candidate_derangement` | 0.312500 | 0.062500 | 0.000000 |
| `candidate_roll` | 0.437500 | -0.062500 | -0.187500 |
| `coefficient_shuffle` | 0.250000 | 0.125000 | 0.000000 |
| `packet_only_source_index` | 0.312500 | 0.062500 | -0.312500 |
| `qwen_substituted_packet` | 0.437500 | -0.062500 | -0.375000 |
| `same_byte_visible_text` | 0.250000 | 0.125000 | -0.125000 |
| `same_source_choice_row_shuffle` | 0.312500 | 0.062500 | -0.250000 |
| `source_rank_control` | 0.312500 | 0.062500 | -0.312500 |
| `source_row_shuffle` | 0.187500 | 0.187500 | 0.062500 |
| `source_score_control` | 0.312500 | 0.062500 | -0.312500 |
| `source_score_quantized_control` | 0.312500 | 0.062500 | -0.312500 |
| `target_decoder_only` | 0.062500 | 0.312500 | 0.062500 |
| `target_derived_packet` | 0.250000 | 0.125000 | -0.095312 |
| `target_only` | 0.250000 | 0.125000 | 0.000000 |
| `top_atom_knockout` | 0.500000 | -0.125000 | -0.312500 |
| `zero_source` | 0.312500 | 0.062500 | -0.125000 |

## No-Op Residual Diagnostics

| Condition | Mean L2 | Mean Ratio | P95 Ratio | Flips vs Target |
|---|---:|---:|---:|---:|
| `atom_shuffle` | 0.263792 | 0.943240 | 0.963716 | 5 |
| `candidate_derangement` | 0.275812 | 0.986221 | 1.000000 | 7 |
| `candidate_roll` | 0.291282 | 1.041535 | 1.000000 | 7 |
| `coefficient_shuffle` | 0.321215 | 1.148566 | 1.066249 | 9 |
| `matched_behavior_atom_decoder_packet` | 0.279666 | 1.000000 | 1.000000 | 0 |
| `same_source_choice_row_shuffle` | 0.278273 | 0.995020 | 0.924171 | 8 |
| `source_row_shuffle` | 0.151603 | 0.542085 | 0.323826 | 1 |
| `target_derived_packet` | 0.144740 | 0.517547 | 0.400352 | 2 |
| `top_atom_knockout` | 0.278542 | 0.995983 | 0.928139 | 7 |
| `zero_source` | 0.087348 | 0.312331 | 0.260309 | 1 |

## Interpretation

This gate tests whether source-hidden innovations become more useful when the sparse atom basis is trained toward target residual behavior rather than unsupervised PCA variance. It passes only if the matched packet beats target-only, target-decoder-only, target-derived packets, same-source-choice and generic wrong-row packets, atom/coefficient destruction, candidate roll/derangement, source-index/rank/score, same-byte text, and Qwen-substitution controls with positive paired uncertainty.
