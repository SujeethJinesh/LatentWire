# Target Self-Resonance HellaSwag Chunk-Encoder Gate

- date: `2026-05-04`
- artifact: `results/target_self_resonance_hellaswag_chunk_encoder_gate_20260504_qwen05_train32_validation32_48`
- pass gate: `False`
- target model: `/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- train/eval rows: `32` / `16`
- prefix tokens: `8`

## Result

- learned encoder agreement: `0.625000`
- learned encoder mean KL: `0.094576`
- chunk-mean agreement/KL: `0.625000` / `0.093210`
- slots-only agreement/KL: `0.625000` / `0.097726`
- best destructive agreement: `0.687500` (`zero_prefix`)
- best destructive mean KL: `0.088720` (`shuffled_chunk_encoder`)

## Condition Metrics

| Condition | Accuracy | Full agreement | Mean KL | Mean margin |
|---|---:|---:|---:|---:|
| `full_prompt` | 0.687500 | 1.000000 | 0.000000 | 0.213110 |
| `chunk_mean_prefix` | 0.500000 | 0.625000 | 0.093210 | -0.096433 |
| `learned_chunk_encoder` | 0.500000 | 0.625000 | 0.094576 | -0.091691 |
| `slots_only_encoder` | 0.375000 | 0.625000 | 0.097726 | -0.118866 |
| `zero_prefix` | 0.500000 | 0.687500 | 0.117645 | -0.218148 |
| `random_same_norm_prefix` | 0.500000 | 0.625000 | 0.139089 | -0.092401 |
| `shuffled_chunk_encoder` | 0.437500 | 0.687500 | 0.088720 | -0.151217 |
| `candidate_derangement` | 0.062500 | 0.125000 | 0.608376 | -1.027210 |

## Interpretation

The held-out target-side encoder does not yet pass. The oracle self-resonance capacity result remains alive, but the current shared chunk-residual encoder is not sufficient.

## Next Gate

Try a stronger query-resampler/ICAE-style encoder or supervised distillation from oracle prefixes before returning to cross-model transfer.
