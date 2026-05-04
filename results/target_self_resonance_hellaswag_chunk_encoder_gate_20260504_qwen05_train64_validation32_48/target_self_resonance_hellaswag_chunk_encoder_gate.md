# Target Self-Resonance HellaSwag Chunk-Encoder Gate

- date: `2026-05-04`
- artifact: `results/target_self_resonance_hellaswag_chunk_encoder_gate_20260504_qwen05_train64_validation32_48`
- pass gate: `True`
- target model: `/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- train/eval rows: `64` / `16`
- prefix tokens: `8`

## Result

- learned encoder agreement: `0.687500`
- learned encoder mean KL: `0.081292`
- chunk-mean agreement/KL: `0.625000` / `0.093210`
- slots-only agreement/KL: `0.687500` / `0.090259`
- best destructive agreement: `0.687500` (`slots_only_encoder`)
- best destructive mean KL: `0.089793` (`shuffled_chunk_encoder`)

## Condition Metrics

| Condition | Accuracy | Full agreement | Mean KL | Mean margin |
|---|---:|---:|---:|---:|
| `full_prompt` | 0.687500 | 1.000000 | 0.000000 | 0.213110 |
| `chunk_mean_prefix` | 0.500000 | 0.625000 | 0.093210 | -0.096433 |
| `learned_chunk_encoder` | 0.437500 | 0.687500 | 0.081292 | -0.068495 |
| `slots_only_encoder` | 0.437500 | 0.687500 | 0.090259 | -0.124051 |
| `zero_prefix` | 0.500000 | 0.687500 | 0.117645 | -0.218148 |
| `random_same_norm_prefix` | 0.500000 | 0.625000 | 0.139089 | -0.092402 |
| `shuffled_chunk_encoder` | 0.500000 | 0.625000 | 0.089793 | -0.172665 |
| `candidate_derangement` | 0.125000 | 0.125000 | 0.623509 | -1.023309 |

## Interpretation

The held-out target-side encoder passes this small gate: it improves the target full-prompt distribution match over raw chunk means and slots-only controls without receiving the context text at target inference time. This promotes the next source-conditioned slot-population gate.

## Next Gate

Add a source-conditioned residual into the same target slots and score source-present minus zero-source-slot, with shuffled-source and candidate-deranged controls.
