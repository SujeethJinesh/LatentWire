# Target-Only Candidate Sampling

- date: `2026-04-27`
- status: `target_only_candidates_sampled`
- git commit: `614f2a2d5482b3d5007b005b4f7df86c23c5479d`
- model: `Qwen/Qwen3-0.6B`
- eval file: `results/svamp32_target_sampling_clean6_20260427/clean6_eval.jsonl`
- samples per example: `16`
- candidate oracle: `2/6`

| Method | Correct | Numeric Coverage |
|---|---:|---:|
| `target_sample_s0` | 0/6 | 6/6 |
| `target_sample_s1` | 0/6 | 6/6 |
| `target_sample_s10` | 1/6 | 6/6 |
| `target_sample_s11` | 0/6 | 6/6 |
| `target_sample_s12` | 0/6 | 6/6 |
| `target_sample_s13` | 1/6 | 6/6 |
| `target_sample_s14` | 1/6 | 6/6 |
| `target_sample_s15` | 1/6 | 6/6 |
| `target_sample_s2` | 0/6 | 6/6 |
| `target_sample_s3` | 1/6 | 6/6 |
| `target_sample_s4` | 1/6 | 6/6 |
| `target_sample_s5` | 1/6 | 6/6 |
| `target_sample_s6` | 1/6 | 6/6 |
| `target_sample_s7` | 1/6 | 6/6 |
| `target_sample_s8` | 1/6 | 6/6 |
| `target_sample_s9` | 1/6 | 6/6 |

## Oracle IDs

`3e8a5691f5443495`, `575d7e83d84c1e67`

## Command

```bash
scripts/sample_target_candidate_surface.py --eval-file results/svamp32_target_sampling_clean6_20260427/clean6_eval.jsonl --model Qwen/Qwen3-0.6B --samples 16 --temperature 0.9 --top-p 0.95 --seed 31 --device mps --dtype float32 --max-new-tokens 64 --use-chat-template --enable-thinking false --output-jsonl results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl --output-json results/svamp32_target_sampling_clean6_20260427/target_only_samples.json --output-md results/svamp32_target_sampling_clean6_20260427/target_only_samples.md
```
