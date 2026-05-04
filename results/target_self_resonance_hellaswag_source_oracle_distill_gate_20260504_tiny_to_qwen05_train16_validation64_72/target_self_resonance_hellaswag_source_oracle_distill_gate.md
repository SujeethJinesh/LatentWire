# Target Self-Resonance HellaSwag Source-Oracle Distill Gate

- date: `2026-05-04`
- artifact: `results/target_self_resonance_hellaswag_source_oracle_distill_gate_20260504_tiny_to_qwen05_train16_validation64_72`
- pass gate: `False`
- source model family: `TinyLlama`
- target model: `/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- train/eval rows: `16` / `8`
- source raw feature fp16 bytes: `16404`
- projected source code fp16 bytes: `32`
- target prefix fp16 bytes: `14336`

## Result

- source-oracle accuracy: `0.125000`
- mean-oracle-slot accuracy: `0.250000`
- source top1/top2 oracle accuracy: `0.500000`
- source-oracle mean KL: `0.118768`
- mean-oracle-slot mean KL: `0.141946`
- best destructive accuracy: `0.375000` (`candidate_derangement`)
- paired CI95 low vs mean oracle slots: `-0.375000`

## Condition Metrics

| Condition | Accuracy | Full agreement | Mean KL | Nonfinite scores |
|---|---:|---:|---:|---:|
| `full_prompt` | 0.500000 | 1.000000 | -0.000000 | 0 |
| `mean_oracle_slots` | 0.250000 | 0.375000 | 0.141946 | 0 |
| `source_oracle_distill_prefix` | 0.125000 | 0.250000 | 0.118768 | 0 |
| `zero_source_code` | 0.250000 | 0.375000 | 0.128425 | 0 |
| `wrong_source_code` | 0.250000 | 0.375000 | 0.125971 | 0 |
| `candidate_roll_source_code` | 0.250000 | 0.375000 | 0.119454 | 0 |
| `target_score_derived_code` | 0.250000 | 0.375000 | 0.107457 | 0 |
| `random_same_norm_prefix` | 0.250000 | 0.375000 | 0.126450 | 0 |
| `source_top1_label_control` | 0.000000 | 0.375000 | 0.145173 | 0 |
| `source_top1_or_top2_oracle` | 0.500000 | 0.625000 | 0.140683 | 0 |
| `candidate_derangement` | 0.375000 | 0.250000 | 0.240177 | 0 |

## Interpretation

The source-oracle distill gate does not pass this held-out slice. If the method ties or loses to wrong-source/zero-source controls, the current projected source code is not causal.

## Next Gate

Inspect row-level helps/harms, then try a stricter target-error-only objective or a smaller source-code ECC/syndrome branch.
