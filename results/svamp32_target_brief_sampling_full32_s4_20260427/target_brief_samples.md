# Candidate Surface Sampling

- date: `2026-04-27`
- status: `candidate_surface_sampled`
- git commit: `4aad04649fd186a985f59fb072e1492a47931d10`
- model: `Qwen/Qwen3-0.6B`
- eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- samples per example: `4`
- candidate oracle: `18/32`

| Method | Correct | Numeric Coverage |
|---|---:|---:|
| `target_brief_sample_s0` | 14/32 | 32/32 |
| `target_brief_sample_s1` | 8/32 | 32/32 |
| `target_brief_sample_s2` | 13/32 | 32/32 |
| `target_brief_sample_s3` | 10/32 | 32/32 |

## Oracle IDs

`0d2b4f7681973e19`, `1d50b408c8f5cd2c`, `2f80204fc9e0896f`, `316b525a4fcbfb89`, `33836927fc9f1a8a`, `41cce6c6e6bb0058`, `47464cc0b064f172`, `4c84ebf42812703b`, `4d780f825bb8541c`, `4ef979ea2bf931df`, `6e9745b37ab6fc45`, `aee922049c757331`, `c042f0a2949ff8e6`, `de1bf4d142544e5b`, `de2a795ab37694af`, `e26bdbcab1449bbe`, `e3ab8666238a289e`, `f367349ce0604296`

## Command

```bash
scripts/sample_target_candidate_surface.py --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl --model Qwen/Qwen3-0.6B --samples 4 --method-prefix target_brief_sample --temperature 0.9 --top-p 0.95 --seed 71 --device mps --dtype float32 --max-new-tokens 64 --prompt-mode source_reasoning --source-reasoning-mode brief_analysis --use-chat-template --enable-thinking false --output-jsonl results/svamp32_target_brief_sampling_full32_s4_20260427/target_brief_samples.jsonl --output-json results/svamp32_target_brief_sampling_full32_s4_20260427/target_brief_samples.json --output-md results/svamp32_target_brief_sampling_full32_s4_20260427/target_brief_samples.md
```
