# ARC Candidate-Alignment Receiver Preflight

- date: `2026-05-02`
- pass gate: `False`
- implementation gate only: `True`
- fit/eval rows: `4` / `4`
- packet bytes: `256B`
- matched accuracy: `0.750`
- best control by accuracy: `target_public_only`
- best control accuracy: `0.500`
- matched margin: `-0.010109`
- best control margin: `0.001987`
- matched minus best-control margin: `-0.012096`

## Conditions

| Condition | Accuracy | Correct / N | Mean Margin |
|---|---:|---:|---:|
| `matched_candidate_alignment_receiver` | 0.750 | 3 / 4 | -0.010109 |
| `target_public_only` | 0.500 | 2 / 4 | 0.001987 |
| `zero_source` | 0.500 | 2 / 4 | 0.001956 |
| `shuffled_source` | 0.500 | 2 / 4 | -0.004655 |
| `same_norm_noise` | 0.250 | 1 / 4 | -0.069005 |
| `train_mean_source` | 0.250 | 1 / 4 | -0.013680 |
| `label_shuffled` | 0.250 | 1 / 4 | -0.010408 |
| `candidate_roll_source` | 0.000 | 0 / 4 | -0.055917 |
| `candidate_derangement` | 0.000 | 0 / 4 | -0.053752 |
| `same_byte_visible_text` | 0.250 | 1 / 4 | -0.500000 |
| `source_label_copy_audit_upper_bound` | 0.750 | 3 / 4 | 500000000.000000 |

## Interpretation

This preflight promotes the candidate-alignment branch only if a matched source candidate sketch beats target-public, erased-source, wrong-row, same-norm-noise, train-mean, label-shuffled, candidate-roll, candidate-derangement, and visible same-byte text controls. A failure weakens this exact external receiver but still leaves richer equivariant rankers alive.

Lay explanation: this experiment gives each answer choice a tiny source-model hint and trains a simple referee to rank the choices. The broken controls shuffle, erase, randomize, or rotate those hints. If the real hints do not beat the broken hints, we have not proven real model-to-model communication.
