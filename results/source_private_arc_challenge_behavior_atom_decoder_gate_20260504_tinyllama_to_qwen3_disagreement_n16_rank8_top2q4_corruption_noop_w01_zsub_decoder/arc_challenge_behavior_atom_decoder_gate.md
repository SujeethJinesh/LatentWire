# ARC-Challenge Behavior-Atom Decoder Packet Gate

- date: `2026-05-04`
- pass gate: `False`
- train/test disagreement rows: `16` / `16`
- matched accuracy: `0.375000`
- target-only accuracy: `0.250000`
- best required control: `top_atom_knockout`
- best required control accuracy: `0.437500`
- worst required paired CI95 low: `-0.375000`
- fired rows: `7`
- helps/harms vs target: `2` / `0`
- packet bytes/row: `7.000`
- same-byte visible-text budget: `7`

## Strict Controls

| Control | Accuracy | Delta | CI95 low |
|---|---:|---:|---:|
| `atom_shuffle` | 0.187500 | 0.187500 | 0.000000 |
| `candidate_derangement` | 0.312500 | 0.062500 | 0.000000 |
| `candidate_roll` | 0.437500 | -0.062500 | -0.187500 |
| `coefficient_shuffle` | 0.312500 | 0.062500 | 0.000000 |
| `packet_only_source_index` | 0.312500 | 0.062500 | -0.312500 |
| `qwen_substituted_packet` | 0.437500 | -0.062500 | -0.375000 |
| `same_byte_visible_text` | 0.250000 | 0.125000 | -0.125000 |
| `same_source_choice_row_shuffle` | 0.375000 | 0.000000 | -0.312500 |
| `source_rank_control` | 0.312500 | 0.062500 | -0.312500 |
| `source_row_shuffle` | 0.187500 | 0.187500 | 0.062500 |
| `source_score_control` | 0.312500 | 0.062500 | -0.312500 |
| `source_score_quantized_control` | 0.312500 | 0.062500 | -0.312500 |
| `target_decoder_only` | 0.062500 | 0.312500 | 0.062500 |
| `target_derived_packet` | 0.187500 | 0.187500 | 0.062500 |
| `target_only` | 0.250000 | 0.125000 | 0.000000 |
| `top_atom_knockout` | 0.437500 | -0.062500 | -0.187500 |
| `zero_source` | 0.250000 | 0.125000 | 0.000000 |

## No-Op Residual Diagnostics

| Condition | Mean L2 | Mean Ratio | P95 Ratio | Flips vs Target |
|---|---:|---:|---:|---:|
| `atom_shuffle` | 0.363048 | 0.997499 | 1.039493 | 5 |
| `candidate_derangement` | 0.355612 | 0.977069 | 1.000000 | 8 |
| `candidate_roll` | 0.385428 | 1.058990 | 1.000000 | 7 |
| `coefficient_shuffle` | 0.435781 | 1.197338 | 1.078428 | 9 |
| `matched_behavior_atom_decoder_packet` | 0.363958 | 1.000000 | 1.000000 | 0 |
| `same_source_choice_row_shuffle` | 0.428530 | 1.177414 | 1.000000 | 7 |
| `source_row_shuffle` | 0.480436 | 1.320031 | 0.561020 | 1 |
| `target_derived_packet` | 0.078915 | 0.216823 | 0.146114 | 1 |
| `top_atom_knockout` | 0.356765 | 0.980237 | 0.921025 | 7 |
| `zero_source` | 0.000000 | 0.000000 | 0.000000 | 0 |

## Interpretation

This gate tests whether source-hidden innovations become more useful when the sparse atom basis is trained toward target residual behavior rather than unsupervised PCA variance. It passes only if the matched packet beats target-only, target-decoder-only, target-derived packets, same-source-choice and generic wrong-row packets, atom/coefficient destruction, candidate roll/derangement, source-index/rank/score, same-byte text, and Qwen-substitution controls with positive paired uncertainty.
