# Target Self-Resonance HellaSwag Consistency-Refined Slot Gate

- date: `2026-05-04`
- artifact: `results/target_self_resonance_hellaswag_consistency_refined_slot_gate_20260504_tiny_to_qwen05_train64_validation88_96`
- pass gate: `False`
- source model family: `TinyLlama`
- target model: `/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- train/eval rows: `64` / `8`
- hidden feature mode: `top2_delta`
- source hidden feature fp16 bytes: `16404`

## Result

- consistency-refined accuracy: `0.375000`
- source residual no-refine accuracy: `0.250000`
- frozen target-slot accuracy: `0.250000`
- best destructive accuracy: `0.375000` (`wrong_source_refine`)
- paired CI95 low vs frozen target slots: `0.000000`

## Condition Metrics

| Condition | Accuracy | Full agreement | Mean KL | Nonfinite scores |
|---|---:|---:|---:|---:|
| `full_prompt` | 0.875000 | 1.000000 | 0.000000 | 0 |
| `frozen_target_slots` | 0.250000 | 0.375000 | 0.176651 | 0 |
| `source_residual_no_refine` | 0.250000 | 0.375000 | 0.178143 | 0 |
| `source_consistency_refined_slots` | 0.375000 | 0.500000 | 0.178246 | 0 |
| `zero_source_refine` | 0.250000 | 0.375000 | 0.176788 | 0 |
| `wrong_source_refine` | 0.375000 | 0.500000 | 0.177839 | 0 |
| `candidate_roll_source_refine` | 0.375000 | 0.500000 | 0.177973 | 0 |
| `target_score_derived_refine` | 0.250000 | 0.375000 | 0.177130 | 0 |
| `refine_step_shuffle` | 0.375000 | 0.500000 | 0.177863 | 0 |
| `random_same_norm_refine` | 0.250000 | 0.375000 | 0.178070 | 0 |
| `source_top1_label_control` | 0.500000 | 0.625000 | 0.106160 | 0 |
| `source_top1_or_top2_oracle` | 1.000000 | 0.875000 | 0.112732 | 0 |
| `candidate_derangement` | 0.125000 | 0.125000 | 0.343127 | 0 |

## Interpretation

The consistency-refined slot gate does not pass this held-out slice. Refinement does not yet turn source evidence into source-specific answer movement under the destructive controls.

## Next Gate

Do not widen this branch until row-level failures reveal a concrete refinement target; the next branch should consider oracle-prefix distillation with source-conditioned features or a smaller shared sparse code.
