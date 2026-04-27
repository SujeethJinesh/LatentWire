# Target-Only Candidate Sampling

- date: `2026-04-27`
- status: `target_only_candidates_sampled`
- git commit: `7fa3f2b35a3dacd1bb789f7a3b2c563e5bb6d45a`
- model: `Qwen/Qwen2.5-Math-1.5B`
- eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- samples per example: `4`
- candidate oracle: `10/32`

| Method | Correct | Numeric Coverage |
|---|---:|---:|
| `target_sample_s0` | 4/32 | 29/32 |
| `target_sample_s1` | 4/32 | 26/32 |
| `target_sample_s2` | 4/32 | 28/32 |
| `target_sample_s3` | 3/32 | 26/32 |

## Oracle IDs

`14bfbfc94f2c2e7b`, `2f80204fc9e0896f`, `3e8a5691f5443495`, `4c84ebf42812703b`, `6e9745b37ab6fc45`, `aee922049c757331`, `b1200c32546a34a5`, `c042f0a2949ff8e6`, `de1bf4d142544e5b`, `f367349ce0604296`

## Command

```bash
scripts/sample_target_candidate_surface.py --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl --model Qwen/Qwen2.5-Math-1.5B --samples 4 --temperature 0.9 --top-p 0.95 --seed 71 --device mps --dtype float32 --max-new-tokens 64 --prompt-mode source_reasoning --source-reasoning-mode brief_analysis --use-chat-template --enable-thinking false --output-jsonl results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl --output-json results/svamp32_source_sampling_full32_s4_20260427/source_samples.json --output-md results/svamp32_source_sampling_full32_s4_20260427/source_samples.md
```
