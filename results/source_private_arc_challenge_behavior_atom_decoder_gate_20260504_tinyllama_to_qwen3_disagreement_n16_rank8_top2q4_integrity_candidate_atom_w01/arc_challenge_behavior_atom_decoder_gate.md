# ARC-Challenge Behavior-Atom Decoder Packet Gate

- date: `2026-05-04`
- pass gate: `False`
- train/test disagreement rows: `16` / `16`
- matched accuracy: `0.250000`
- target-only accuracy: `0.250000`
- best required control: `qwen_substituted_packet`
- best required control accuracy: `0.437500`
- worst required paired CI95 low: `-0.437500`
- fired rows: `0`
- helps/harms vs target: `0` / `0`
- packet bytes/row: `7.000`
- same-byte visible-text budget: `7`

## Strict Controls

| Control | Accuracy | Delta | CI95 low |
|---|---:|---:|---:|
| `target_only` | 0.250000 | 0.000000 | 0.000000 |
| `target_decoder_only` | 0.062500 | 0.187500 | -0.062500 |
| `target_derived_packet` | 0.250000 | 0.000000 | 0.000000 |
| `zero_source` | 0.250000 | 0.000000 | 0.000000 |
| `source_row_shuffle` | 0.250000 | 0.000000 | 0.000000 |
| `same_source_choice_row_shuffle` | 0.250000 | 0.000000 | 0.000000 |
| `atom_shuffle` | 0.250000 | 0.000000 | 0.000000 |
| `coefficient_shuffle` | 0.250000 | 0.000000 | 0.000000 |
| `top_atom_knockout` | 0.250000 | 0.000000 | 0.000000 |
| `candidate_roll` | 0.250000 | 0.000000 | 0.000000 |
| `candidate_derangement` | 0.250000 | 0.000000 | 0.000000 |
| `packet_only_source_index` | 0.312500 | -0.062500 | -0.407812 |
| `source_rank_control` | 0.312500 | -0.062500 | -0.375000 |
| `source_score_control` | 0.312500 | -0.062500 | -0.375000 |
| `source_score_quantized_control` | 0.312500 | -0.062500 | -0.345312 |
| `same_byte_visible_text` | 0.250000 | 0.000000 | -0.187500 |
| `qwen_substituted_packet` | 0.437500 | -0.187500 | -0.437500 |

## No-Op Residual Diagnostics

| Condition | Mean L2 | Mean Ratio | P95 Ratio | Flips vs Target |
|---|---:|---:|---:|---:|
| `matched_behavior_atom_decoder_packet` | 0.000000 | 1.000000 | 1.000000 | 0 |
| `target_derived_packet` | 0.000000 | 0.000000 | 0.000000 | 0 |
| `zero_source` | 0.000000 | 0.000000 | 0.000000 | 0 |
| `source_row_shuffle` | 0.000000 | 0.000000 | 0.000000 | 0 |
| `same_source_choice_row_shuffle` | 0.000000 | 0.000000 | 0.000000 | 0 |
| `atom_shuffle` | 0.033654 | 33653791736.305611 | 134615166945.222443 | 1 |
| `coefficient_shuffle` | 0.000000 | 0.000000 | 0.000000 | 0 |
| `top_atom_knockout` | 0.000000 | 0.000000 | 0.000000 | 0 |
| `candidate_roll` | 0.000000 | 0.000000 | 0.000000 | 0 |
| `candidate_derangement` | 0.000000 | 0.000000 | 0.000000 | 0 |

## Interpretation

This gate tests whether source-hidden innovations become more useful when the sparse atom basis is trained toward target residual behavior rather than unsupervised PCA variance. It passes only if the matched packet beats target-only, target-decoder-only, target-derived packets, same-source-choice and generic wrong-row packets, atom/coefficient destruction, candidate roll/derangement, source-index/rank/score, same-byte text, and Qwen-substitution controls with positive paired uncertainty.
