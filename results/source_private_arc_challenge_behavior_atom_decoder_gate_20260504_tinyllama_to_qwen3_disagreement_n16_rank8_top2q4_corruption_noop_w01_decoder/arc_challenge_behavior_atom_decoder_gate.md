# ARC-Challenge Behavior-Atom Decoder Packet Gate

- date: `2026-05-04`
- pass gate: `False`
- train/test disagreement rows: `16` / `16`
- matched accuracy: `0.437500`
- target-only accuracy: `0.250000`
- best required control: `candidate_roll`
- best required control accuracy: `0.500000`
- worst required paired CI95 low: `-0.312500`
- fired rows: `7`
- helps/harms vs target: `3` / `0`
- packet bytes/row: `7.000`
- same-byte visible-text budget: `7`

## Strict Controls

| Control | Accuracy | Delta | CI95 low |
|---|---:|---:|---:|
| `atom_shuffle` | 0.125000 | 0.312500 | 0.029688 |
| `candidate_derangement` | 0.375000 | 0.062500 | 0.000000 |
| `candidate_roll` | 0.500000 | -0.062500 | -0.187500 |
| `coefficient_shuffle` | 0.187500 | 0.250000 | 0.062500 |
| `packet_only_source_index` | 0.312500 | 0.125000 | -0.250000 |
| `qwen_substituted_packet` | 0.437500 | 0.000000 | -0.312500 |
| `same_byte_visible_text` | 0.250000 | 0.187500 | -0.062500 |
| `same_source_choice_row_shuffle` | 0.375000 | 0.062500 | -0.250000 |
| `source_rank_control` | 0.312500 | 0.125000 | -0.187500 |
| `source_row_shuffle` | 0.187500 | 0.250000 | 0.062500 |
| `source_score_control` | 0.312500 | 0.125000 | -0.250000 |
| `source_score_quantized_control` | 0.312500 | 0.125000 | -0.187500 |
| `target_decoder_only` | 0.062500 | 0.375000 | 0.092188 |
| `target_derived_packet` | 0.125000 | 0.312500 | -0.062500 |
| `target_only` | 0.250000 | 0.187500 | 0.000000 |
| `top_atom_knockout` | 0.437500 | 0.000000 | -0.187500 |
| `zero_source` | 0.312500 | 0.125000 | -0.062500 |

## No-Op Residual Diagnostics

| Condition | Mean L2 | Mean Ratio | P95 Ratio | Flips vs Target |
|---|---:|---:|---:|---:|
| `atom_shuffle` | 0.380669 | 0.969258 | 0.921560 | 7 |
| `candidate_derangement` | 0.393133 | 1.000995 | 1.000000 | 8 |
| `candidate_roll` | 0.421151 | 1.072334 | 1.000000 | 8 |
| `coefficient_shuffle` | 0.448445 | 1.141828 | 1.007286 | 10 |
| `matched_behavior_atom_decoder_packet` | 0.392743 | 1.000000 | 1.000000 | 0 |
| `same_source_choice_row_shuffle` | 0.432633 | 1.101569 | 0.895828 | 9 |
| `source_row_shuffle` | 0.378784 | 0.964458 | 0.484751 | 1 |
| `target_derived_packet` | 0.214260 | 0.545549 | 0.416931 | 11 |
| `top_atom_knockout` | 0.407937 | 1.038689 | 0.926853 | 8 |
| `zero_source` | 0.141370 | 0.359957 | 0.299250 | 1 |

## Interpretation

This gate tests whether source-hidden innovations become more useful when the sparse atom basis is trained toward target residual behavior rather than unsupervised PCA variance. It passes only if the matched packet beats target-only, target-decoder-only, target-derived packets, same-source-choice and generic wrong-row packets, atom/coefficient destruction, candidate roll/derangement, source-index/rank/score, same-byte text, and Qwen-substitution controls with positive paired uncertainty.
