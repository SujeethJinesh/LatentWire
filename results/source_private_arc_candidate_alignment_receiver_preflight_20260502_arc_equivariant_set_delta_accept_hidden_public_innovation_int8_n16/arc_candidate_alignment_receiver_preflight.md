# ARC Candidate-Alignment Receiver Preflight

- date: `2026-05-02`
- pass gate: `False`
- implementation gate only: `True`
- fit/eval rows: `8` / `8`
- packet bytes: `66B`
- matched accuracy: `0.375`
- best control by accuracy: `target_public_only`
- best control accuracy: `0.375`
- matched margin: `-0.003784`
- best control margin: `-0.003784`
- matched minus best-control margin: `0.000000`
- zero-source exact target-public match: `True`

## Conditions

| Condition | Accuracy | Correct / N | Mean Margin |
|---|---:|---:|---:|
| `matched_candidate_alignment_receiver` | 0.375 | 3 / 8 | -0.003784 |
| `target_public_only` | 0.375 | 3 / 8 | -0.003784 |
| `zero_source` | 0.375 | 3 / 8 | -0.003784 |
| `shuffled_source` | 0.375 | 3 / 8 | -0.003784 |
| `same_norm_noise` | 0.375 | 3 / 8 | -0.003784 |
| `train_mean_source` | 0.375 | 3 / 8 | -0.003784 |
| `target_derived_source` | 0.375 | 3 / 8 | -0.003784 |
| `label_shuffled` | 0.375 | 3 / 8 | -0.003784 |
| `candidate_roll_source` | 0.375 | 3 / 8 | -0.003784 |
| `candidate_derangement` | 0.250 | 2 / 8 | -0.009751 |
| `same_byte_visible_text` | 0.250 | 2 / 8 | -0.750000 |
| `source_label_copy_audit_upper_bound` | 0.500 | 4 / 8 | 0.000000 |

## Interpretation

This preflight promotes the candidate-alignment branch only if a matched source candidate sketch beats target-public, erased-source, wrong-row, same-norm-noise, train-mean, label-shuffled, candidate-roll, candidate-derangement, and visible same-byte text controls. A failure weakens this exact external receiver but still leaves richer equivariant rankers alive.

Lay explanation: this experiment gives each answer choice a tiny source-model hint and trains a simple referee to rank the choices. The broken controls shuffle, erase, randomize, or rotate those hints. If the real hints do not beat the broken hints, we have not proven real model-to-model communication.
