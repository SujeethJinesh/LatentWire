# Target Self-Resonance HellaSwag Oracle-Distill Gate

- date: `2026-05-04`
- artifact: `results/target_self_resonance_hellaswag_oracle_distill_gate_20260504_qwen05_train16_validation64_72`
- pass gate: `False`
- target model: `/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- train/eval rows: `16` / `8`
- prefix tokens: `8`

## Result

- distilled encoder agreement: `0.375000`
- distilled encoder mean KL: `0.128162`
- chunk-mean agreement/KL: `0.375000` / `0.105277`
- slots-only agreement/KL: `0.250000` / `0.106234`
- KL gain vs best target-only control: `-0.022885`

## Condition Metrics

| Condition | Accuracy | Full agreement | Mean KL | Nonfinite scores |
|---|---:|---:|---:|---:|
| `full_prompt` | 0.500000 | 1.000000 | -0.000000 | 0 |
| `chunk_mean_prefix` | 0.250000 | 0.375000 | 0.105277 | 0 |
| `oracle_distill_encoder` | 0.250000 | 0.375000 | 0.128162 | 0 |
| `slots_only_oracle_distill` | 0.125000 | 0.250000 | 0.106234 | 0 |
| `zero_prefix` | 0.250000 | 0.375000 | 0.135393 | 0 |
| `random_same_norm_prefix` | 0.125000 | 0.250000 | 0.126130 | 0 |
| `shuffled_oracle_distill` | 0.250000 | 0.375000 | 0.127895 | 0 |
| `candidate_derangement` | 0.250000 | 0.375000 | 0.262669 | 0 |

## Interpretation

Oracle-prefix distillation does not yet pass the held-out target-only controls. The oracle capacity result remains alive, but the current distilled encoder is not yet the positive method.

## Next Gate

Increase the train-oracle slice or switch to a query-resampler/ICAE-style encoder before adding source-conditioned transfer.
