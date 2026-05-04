# Target Self-Resonance HellaSwag Soft-Prefix Gate

- date: `2026-05-04`
- artifact: `results/target_self_resonance_hellaswag_soft_prefix_gate_20260504_qwen05_validation0_16`
- target model: `/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- rows: `16`
- prefix tokens: `8`
- pass gate: `True`

## Result

- full-prompt accuracy: `0.250000`
- optimized-prefix accuracy: `0.312500`
- optimized-prefix agreement with full prompt: `0.937500`
- optimized-prefix mean KL to full prompt: `0.000853`
- chunk-mean prefix agreement: `0.500000`
- chunk-mean prefix mean KL: `0.126353`
- optimized KL gain vs chunk-mean: `0.125500`
- best destructive-control agreement: `0.625000` (`shuffled_optimized_prefix`)
- best destructive-control mean KL: `0.153753` (`shuffled_optimized_prefix`)
- optimized KL gain vs best destructive: `0.152900`

## Condition Metrics

| Condition | Accuracy | Full agreement | Mean KL | Mean margin |
|---|---:|---:|---:|---:|
| `full_prompt` | 0.250000 | 1.000000 | -0.000000 | -0.679028 |
| `chunk_mean_prefix` | 0.250000 | 0.500000 | 0.126353 | -0.737623 |
| `optimized_soft_prefix` | 0.312500 | 0.937500 | 0.000853 | -0.644305 |
| `zero_prefix` | 0.125000 | 0.437500 | 0.155816 | -0.840962 |
| `random_same_norm_prefix` | 0.250000 | 0.375000 | 0.154319 | -0.676915 |
| `shuffled_optimized_prefix` | 0.187500 | 0.625000 | 0.153753 | -0.699780 |
| `candidate_derangement` | 0.375000 | 0.062500 | 0.581183 | -0.553549 |

## Interpretation

The target-self branch stays alive if optimized soft prefixes preserve the frozen target model's full-context choice behavior more reliably than chunk-mean, zero, random, shuffled, and candidate-deranged controls. This does not yet prove a source-private encoder; it only tests whether the target has a compact continuous control surface worth learning into.

## Next Gate

Train a generalizing text-to-prefix encoder on official train rows and evaluate held-out rows with the same destructive controls; then repeat with Phi as target if Mac runtime is tolerable.
