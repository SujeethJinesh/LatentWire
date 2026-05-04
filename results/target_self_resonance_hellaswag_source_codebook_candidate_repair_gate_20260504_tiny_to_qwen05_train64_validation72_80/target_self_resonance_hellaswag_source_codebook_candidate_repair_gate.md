# Target Self-Resonance HellaSwag Source-Codebook Candidate Repair Gate

- date: `2026-05-04`
- artifact: `results/target_self_resonance_hellaswag_source_codebook_candidate_repair_gate_20260504_tiny_to_qwen05_train64_validation72_80`
- pass gate: `False`
- selected code mode: `top1_top2_margin_entropy`
- selected prior/delta weights: `3.0` / `1.5`
- source packet raw/framed bytes: `1` / `4`

## Result

- source-codebook accuracy: `0.500000`
- frozen target-slot accuracy: `0.375000`
- source-top1 label-control accuracy: `0.750000`
- source-pair oracle accuracy: `1.000000`
- paired CI95 low vs frozen target slots: `0.000000`
- paired CI95 low vs source-top1: `-0.750000`

## Condition Metrics

| Condition | Accuracy | Full agreement | Mean KL |
|---|---:|---:|---:|
| `full_prompt` | 0.750000 | 1.000000 | -0.000000 |
| `frozen_target_slots` | 0.375000 | 0.500000 | 0.387614 |
| `source_codebook_repair` | 0.500000 | 0.500000 | 0.419073 |
| `zero_source_codebook` | 0.500000 | 0.500000 | 0.470544 |
| `wrong_source_codebook` | 0.375000 | 0.375000 | 0.757971 |
| `candidate_roll_source_codebook` | 0.625000 | 0.500000 | 0.518162 |
| `target_derived_codebook` | 0.375000 | 0.500000 | 0.631026 |
| `random_codebook` | 0.125000 | 0.125000 | 2.061684 |
| `source_top1_label_control` | 0.750000 | 0.750000 | 0.124459 |
| `source_top2_label_control` | 0.250000 | 0.250000 | 0.423126 |
| `source_top1_or_top2_oracle` | 1.000000 | 0.750000 | 0.137125 |
| `candidate_derangement` | 0.000000 | 0.125000 | 0.946902 |

## Interpretation

The quantized source-codebook candidate repair gate does not pass this held-out slice. It is useful as a direct test of whether a tiny source code can beat target-only and source-copy shortcuts rather than merely improving KL.

## Next Gate

Inspect whether the source top-1/top-2 oracle has enough headroom; if yes, add a calibrated router that chooses between source top-1, source top-2, and target-local evidence under the same controls.
