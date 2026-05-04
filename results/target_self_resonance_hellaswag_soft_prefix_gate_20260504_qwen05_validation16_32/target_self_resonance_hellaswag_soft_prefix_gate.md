# Target Self-Resonance HellaSwag Soft-Prefix Gate

- date: `2026-05-04`
- artifact: `results/target_self_resonance_hellaswag_soft_prefix_gate_20260504_qwen05_validation16_32`
- target model: `/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- rows: `16`
- prefix tokens: `8`
- pass gate: `True`

## Result

- full-prompt accuracy: `0.562500`
- optimized-prefix accuracy: `0.375000`
- optimized-prefix agreement with full prompt: `0.812500`
- optimized-prefix mean KL to full prompt: `0.013037`
- chunk-mean prefix agreement: `0.375000`
- chunk-mean prefix mean KL: `0.131362`
- optimized KL gain vs chunk-mean: `0.118326`
- best destructive-control agreement: `0.500000` (`zero_prefix`)
- best destructive-control mean KL: `0.135498` (`random_same_norm_prefix`)
- optimized KL gain vs best destructive: `0.122461`

## Condition Metrics

| Condition | Accuracy | Full agreement | Mean KL | Mean margin |
|---|---:|---:|---:|---:|
| `full_prompt` | 0.562500 | 1.000000 | -0.000000 | -0.057494 |
| `chunk_mean_prefix` | 0.250000 | 0.375000 | 0.131362 | -0.604936 |
| `optimized_soft_prefix` | 0.375000 | 0.812500 | 0.013037 | -0.194779 |
| `zero_prefix` | 0.375000 | 0.500000 | 0.186930 | -0.861054 |
| `random_same_norm_prefix` | 0.250000 | 0.500000 | 0.135498 | -0.634341 |
| `shuffled_optimized_prefix` | 0.312500 | 0.500000 | 0.188366 | -0.605139 |
| `candidate_derangement` | 0.375000 | 0.125000 | 0.397866 | -0.475744 |

## Interpretation

The target-self branch stays alive if optimized soft prefixes preserve the frozen target model's full-context choice behavior more reliably than chunk-mean, zero, random, shuffled, and candidate-deranged controls. This does not yet prove a source-private encoder; it only tests whether the target has a compact continuous control surface worth learning into.

## Next Gate

Train a generalizing text-to-prefix encoder on official train rows and evaluate held-out rows with the same destructive controls; then repeat with Phi as target if Mac runtime is tolerable.
