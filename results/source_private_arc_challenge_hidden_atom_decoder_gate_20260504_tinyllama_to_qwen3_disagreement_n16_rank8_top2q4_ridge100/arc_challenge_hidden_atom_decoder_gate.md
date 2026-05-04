# ARC-Challenge Hidden-Atom Decoder Packet Gate

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

## Strict Controls

| Control | Accuracy | Delta | CI95 low |
|---|---:|---:|---:|
| `target_only` | 0.250000 | 0.000000 | 0.000000 |
| `target_decoder_only` | 0.250000 | 0.000000 | 0.000000 |
| `target_derived_packet` | 0.250000 | 0.000000 | 0.000000 |
| `zero_source` | 0.250000 | 0.000000 | 0.000000 |
| `source_row_shuffle` | 0.187500 | 0.062500 | 0.000000 |
| `atom_shuffle` | 0.250000 | 0.000000 | 0.000000 |
| `coefficient_shuffle` | 0.250000 | 0.000000 | 0.000000 |
| `top_atom_knockout` | 0.250000 | 0.000000 | 0.000000 |
| `candidate_roll` | 0.250000 | 0.000000 | 0.000000 |
| `candidate_derangement` | 0.250000 | 0.000000 | 0.000000 |
| `packet_only_source_index` | 0.312500 | -0.062500 | -0.375000 |
| `source_rank_control` | 0.312500 | -0.062500 | -0.375000 |
| `source_score_control` | 0.312500 | -0.062500 | -0.375000 |
| `source_score_quantized_control` | 0.312500 | -0.062500 | -0.407812 |
| `same_byte_visible_text` | 0.312500 | -0.062500 | -0.375000 |
| `qwen_substituted_packet` | 0.437500 | -0.187500 | -0.437500 |

## Interpretation

This gate tests whether compact source-hidden PCA atom packets become useful when decoded by a target-conditioned residual receiver instead of direct soft-prefix reconstruction. It passes only if the matched packet beats target-only, target-decoder-only, target-derived packets, source-row/atom/coefficient destruction, candidate roll/derangement, source-index/rank/score, same-byte text, and Qwen-substitution controls with positive paired uncertainty.
