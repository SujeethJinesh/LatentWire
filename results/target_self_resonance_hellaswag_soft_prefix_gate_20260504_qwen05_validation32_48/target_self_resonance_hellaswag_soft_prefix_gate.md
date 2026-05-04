# Target Self-Resonance HellaSwag Soft-Prefix Gate

- date: `2026-05-04`
- artifact: `results/target_self_resonance_hellaswag_soft_prefix_gate_20260504_qwen05_validation32_48`
- target model: `/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- rows: `16`
- prefix tokens: `8`
- pass gate: `True`

## Result

- full-prompt accuracy: `0.687500`
- optimized-prefix accuracy: `0.687500`
- optimized-prefix agreement with full prompt: `1.000000`
- optimized-prefix mean KL to full prompt: `0.000060`
- chunk-mean prefix agreement: `0.625000`
- chunk-mean prefix mean KL: `0.093210`
- optimized KL gain vs chunk-mean: `0.093150`
- best destructive-control agreement: `0.687500` (`zero_prefix`)
- best destructive-control mean KL: `0.098435` (`shuffled_optimized_prefix`)
- optimized KL gain vs best destructive: `0.098375`

## Condition Metrics

| Condition | Accuracy | Full agreement | Mean KL | Mean margin |
|---|---:|---:|---:|---:|
| `full_prompt` | 0.687500 | 1.000000 | 0.000000 | 0.213110 |
| `chunk_mean_prefix` | 0.500000 | 0.625000 | 0.093210 | -0.096433 |
| `optimized_soft_prefix` | 0.687500 | 1.000000 | 0.000060 | 0.214708 |
| `zero_prefix` | 0.500000 | 0.687500 | 0.117645 | -0.218148 |
| `random_same_norm_prefix` | 0.500000 | 0.625000 | 0.109253 | -0.189810 |
| `shuffled_optimized_prefix` | 0.375000 | 0.625000 | 0.098435 | -0.308707 |
| `candidate_derangement` | 0.062500 | 0.000000 | 0.737876 | -1.351056 |

## Interpretation

The target-self branch stays alive if optimized soft prefixes preserve the frozen target model's full-context choice behavior more reliably than chunk-mean, zero, random, shuffled, and candidate-deranged controls. This does not yet prove a source-private encoder; it only tests whether the target has a compact continuous control surface worth learning into.

## Next Gate

Train a generalizing text-to-prefix encoder on official train rows and evaluate held-out rows with the same destructive controls; then repeat with Phi as target if Mac runtime is tolerable.
