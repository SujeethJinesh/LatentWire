# Target Self-Resonance HellaSwag Chunk-Encoder Gate

- date: `2026-05-04`
- artifact: `results/target_self_resonance_hellaswag_chunk_encoder_gate_20260504_qwen05_train64_validation48_64`
- pass gate: `False`
- target model: `/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- train/eval rows: `64` / `16`
- prefix tokens: `8`

## Result

- learned encoder agreement: `0.750000`
- learned encoder mean KL: `0.074528`
- chunk-mean agreement/KL: `0.687500` / `0.075388`
- slots-only agreement/KL: `0.687500` / `0.067913`
- best destructive agreement: `0.750000` (`zero_prefix`)
- best destructive mean KL: `0.067913` (`slots_only_encoder`)

## Condition Metrics

| Condition | Accuracy | Full agreement | Mean KL | Mean margin |
|---|---:|---:|---:|---:|
| `full_prompt` | 0.312500 | 1.000000 | 0.000000 | -0.341518 |
| `chunk_mean_prefix` | 0.250000 | 0.687500 | 0.075388 | -0.488470 |
| `learned_chunk_encoder` | 0.250000 | 0.750000 | 0.074528 | -0.451213 |
| `slots_only_encoder` | 0.250000 | 0.687500 | 0.067913 | -0.411399 |
| `zero_prefix` | 0.250000 | 0.750000 | 0.108633 | -0.609119 |
| `random_same_norm_prefix` | 0.250000 | 0.687500 | 0.114336 | -0.655063 |
| `shuffled_chunk_encoder` | 0.250000 | 0.687500 | 0.075647 | -0.394925 |
| `candidate_derangement` | 0.250000 | 0.125000 | 0.525860 | -0.849198 |

## Interpretation

The held-out target-side encoder does not yet pass. The oracle self-resonance capacity result remains alive, but the current shared chunk-residual encoder is not sufficient.

## Next Gate

Try a stronger query-resampler/ICAE-style encoder or supervised distillation from oracle prefixes before returning to cross-model transfer.
