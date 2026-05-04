# Target Self-Resonance HellaSwag Query-Resampler Gate

- date: `2026-05-04`
- artifact: `results/target_self_resonance_hellaswag_query_resampler_gate_20260504_qwen05_train64_validation72_80_contrastive`
- pass gate: `False`
- target model: `/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- train/eval rows: `64` / `8`
- prefix tokens: `8`

## Result

- query-resampler agreement: `0.500000`
- query-resampler mean KL: `0.557950`
- chunk-mean agreement/KL: `0.625000` / `0.426502`
- slots-only agreement/KL: `0.500000` / `0.217971`
- KL gain vs best control: `-0.339979`
- mean normalized attention entropy: `1.000000`

## Condition Metrics

| Condition | Accuracy | Full agreement | Mean KL | Nonfinite scores |
|---|---:|---:|---:|---:|
| `full_prompt` | 0.750000 | 1.000000 | -0.000000 | 0 |
| `chunk_mean_prefix` | 0.500000 | 0.625000 | 0.426502 | 0 |
| `query_resampler` | 0.375000 | 0.500000 | 0.557950 | 0 |
| `slots_only_query` | 0.375000 | 0.500000 | 0.217971 | 0 |
| `zero_prefix` | 0.375000 | 0.500000 | 0.557950 | 0 |
| `random_same_norm_prefix` | 0.375000 | 0.500000 | 0.557949 | 0 |
| `shuffled_query_resampler` | 0.375000 | 0.500000 | 0.557950 | 0 |
| `candidate_derangement` | 0.000000 | 0.000000 | 0.819783 | 0 |

## Interpretation

The query-resampler target interface does not pass this held-out gate. Treat this as a negative target-only compression ablation unless a future source-conditioned residual packet beats slots-only and wrong-row controls.

## Next Gate

Implement source-conditioned residual slots over a frozen target-slot baseline with zero-source, wrong-source, target-derived-code, and candidate-deranged controls.
