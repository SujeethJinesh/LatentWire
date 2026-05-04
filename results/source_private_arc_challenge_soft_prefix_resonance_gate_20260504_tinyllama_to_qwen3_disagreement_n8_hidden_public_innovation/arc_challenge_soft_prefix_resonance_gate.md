# ARC-Challenge Soft-Prefix Resonance Gate

- date: `2026-05-04`
- pass gate: `False`
- train/test disagreement rows: `8` / `8`
- matched accuracy: `0.500000`
- best required control: `qwen_substituted_packet`
- best required control accuracy: `0.625000`
- worst required paired CI95 low: `-0.625000`

## Strict Controls

| Control | Accuracy | Delta | CI95 low |
|---|---:|---:|---:|
| `target_only` | 0.500000 | 0.000000 | -0.375000 |
| `slots_only_prefix` | 0.500000 | 0.000000 | -0.375000 |
| `zero_source` | 0.250000 | 0.250000 | 0.000000 |
| `source_row_shuffle` | 0.250000 | 0.250000 | -0.250000 |
| `candidate_roll` | 0.500000 | 0.000000 | 0.000000 |
| `target_derived_prefix` | 0.500000 | 0.000000 | -0.375000 |
| `packet_only_source_index` | 0.125000 | 0.375000 | 0.000000 |
| `source_rank_control` | 0.125000 | 0.375000 | 0.125000 |
| `source_score_control` | 0.125000 | 0.375000 | 0.125000 |
| `same_byte_visible_text` | 0.500000 | 0.000000 | -0.375000 |
| `qwen_substituted_packet` | 0.625000 | -0.125000 | -0.625000 |
| `candidate_derangement` | 0.250000 | 0.250000 | -0.250000 |

## Interpretation

This strict gate passes only if a validation-trained source-conditioned soft prefix beats every required target-only, source-destroying, same-byte, source-index/rank/score, and Qwen-substitution control on frozen test disagreement rows with positive paired uncertainty.

Lay explanation: this trains a small translator on validation questions where the source and target disagree, then tests once on new disagreement questions. The target gets hidden soft-prefix tokens, not the source text. The controls ask whether the same result comes from target-only memory, a row shuffle, candidate shuffle, raw source rank/score shortcuts, or visible same-byte text.
