# Target-Only Candidate Sampling

- date: `2026-04-27`
- status: `target_only_candidates_sampled`
- git commit: `76abab281d43ec30e509d58e750c3232f2adcb0e`
- model: `Qwen/Qwen3-0.6B`
- eval file: `results/target_only_sampling_clean3_20260427/clean_source_only_eval.jsonl`
- samples per example: `8`
- candidate oracle: `1/3`

| Method | Correct | Numeric Coverage |
|---|---:|---:|
| `target_sample_s0` | 0/3 | 3/3 |
| `target_sample_s1` | 1/3 | 3/3 |
| `target_sample_s2` | 1/3 | 3/3 |
| `target_sample_s3` | 0/3 | 3/3 |
| `target_sample_s4` | 1/3 | 3/3 |
| `target_sample_s5` | 0/3 | 3/3 |
| `target_sample_s6` | 0/3 | 3/3 |
| `target_sample_s7` | 0/3 | 3/3 |

## Oracle IDs

`14bfbfc94f2c2e7b`

## Command

```bash
scripts/sample_target_candidate_surface.py --eval-file results/target_only_sampling_clean3_20260427/clean_source_only_eval.jsonl --model Qwen/Qwen3-0.6B --samples 8 --temperature 0.9 --top-p 0.95 --seed 11 --device cpu --dtype float32 --max-new-tokens 96 --use-chat-template --enable-thinking false --output-jsonl results/target_only_sampling_clean3_20260427/target_only_samples.jsonl --output-json results/target_only_sampling_clean3_20260427/target_only_samples.json --output-md results/target_only_sampling_clean3_20260427/target_only_samples.md
```
