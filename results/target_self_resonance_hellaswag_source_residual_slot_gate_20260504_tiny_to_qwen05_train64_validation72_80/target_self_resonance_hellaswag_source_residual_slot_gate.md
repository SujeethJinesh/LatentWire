# Target Self-Resonance HellaSwag Source-Residual Slot Gate

- date: `2026-05-04`
- artifact: `results/target_self_resonance_hellaswag_source_residual_slot_gate_20260504_tiny_to_qwen05_train64_validation72_80`
- pass gate: `False`
- source model family: `TinyLlama`
- target model: `/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- train/eval rows: `64` / `8`
- source feature mode: `top2_margin`
- source packet raw/framed bytes: `2` / `5`

## Result

- source-residual accuracy: `0.375000`
- frozen target-slot accuracy: `0.375000`
- source-top1 label-control accuracy: `0.750000`
- source-residual mean KL: `0.324309`
- frozen target-slot mean KL: `0.380034`
- best destructive accuracy: `0.750000` (`source_top1_label_control`)
- paired CI95 low vs frozen target slots: `0.000000`

## Condition Metrics

| Condition | Accuracy | Full agreement | Mean KL | Nonfinite scores |
|---|---:|---:|---:|---:|
| `full_prompt` | 0.750000 | 1.000000 | -0.000000 | 0 |
| `frozen_target_slots` | 0.375000 | 0.500000 | 0.380034 | 0 |
| `source_residual_slots` | 0.375000 | 0.500000 | 0.324309 | 0 |
| `zero_source_residual` | 0.375000 | 0.500000 | 0.363328 | 0 |
| `wrong_source_residual` | 0.375000 | 0.500000 | 0.350110 | 0 |
| `candidate_roll_source_residual` | 0.375000 | 0.500000 | 0.342539 | 0 |
| `target_derived_residual` | 0.375000 | 0.500000 | 0.337038 | 0 |
| `random_same_norm_residual` | 0.375000 | 0.500000 | 0.383617 | 0 |
| `source_top1_label_control` | 0.750000 | 0.750000 | 0.124459 | 0 |
| `candidate_derangement` | 0.000000 | 0.000000 | 0.632338 | 0 |

## Interpretation

The source-conditioned residual-slot gate does not pass this held-out slice. The result is still informative because it directly tests source-present residuals against zero-source, wrong-source, target-derived, and label-copy controls.

## Next Gate

Inspect row-level failures and either add a quantized source-conditioned candidate repair head or move to a stronger residual-slot codebook/denoising interface.
