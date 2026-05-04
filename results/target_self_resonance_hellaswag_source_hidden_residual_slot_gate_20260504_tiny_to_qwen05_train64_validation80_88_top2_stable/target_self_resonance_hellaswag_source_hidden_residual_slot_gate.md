# Target Self-Resonance HellaSwag Source-Hidden Residual Slot Gate

- date: `2026-05-04`
- artifact: `results/target_self_resonance_hellaswag_source_hidden_residual_slot_gate_20260504_tiny_to_qwen05_train64_validation80_88_top2_stable`
- pass gate: `False`
- source model family: `TinyLlama`
- target model: `/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- train/eval rows: `64` / `8`
- hidden feature mode: `top2_delta`
- source hidden feature fp16 bytes: `16404`
- target prefix fp16 bytes: `14336`

## Result

- source-hidden residual accuracy: `0.375000`
- frozen target-slot accuracy: `0.375000`
- source-top1 label-control accuracy: `0.125000`
- source top1/top2 oracle accuracy: `0.625000`
- source-hidden residual mean KL: `0.165418`
- frozen target-slot mean KL: `0.166265`
- best destructive accuracy: `0.375000` (`zero_source_hidden`)
- paired CI95 low vs frozen target slots: `0.000000`

## Condition Metrics

| Condition | Accuracy | Full agreement | Mean KL | Nonfinite scores |
|---|---:|---:|---:|---:|
| `full_prompt` | 0.250000 | 1.000000 | -0.000000 | 0 |
| `frozen_target_slots` | 0.375000 | 0.375000 | 0.166265 | 0 |
| `source_hidden_residual_slots` | 0.375000 | 0.375000 | 0.165418 | 0 |
| `zero_source_hidden` | 0.375000 | 0.375000 | 0.166258 | 0 |
| `wrong_source_hidden` | 0.375000 | 0.375000 | 0.165565 | 0 |
| `candidate_roll_source_hidden` | 0.375000 | 0.375000 | 0.165110 | 0 |
| `target_score_derived_hidden_template` | 0.375000 | 0.375000 | 0.166113 | 0 |
| `random_same_norm_residual` | 0.375000 | 0.375000 | 0.166934 | 0 |
| `source_top1_label_control` | 0.125000 | 0.750000 | 0.083596 | 0 |
| `source_top1_or_top2_oracle` | 0.625000 | 0.500000 | 0.163914 | 0 |
| `candidate_derangement` | 0.375000 | 0.250000 | 0.197878 | 0 |

## Interpretation

The source-hidden residual-slot gate does not pass this held-out slice. This weakens the direct hidden-to-target-soft-slot branch unless row-level failures show a clear fix.

## Next Gate

Analyze row-level failures, then decide between a smaller SAE/PCA hidden code, an oracle-prefix distillation target, or a consistency-refined target-native slot interface.
