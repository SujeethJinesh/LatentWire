# Target-Only Candidate Sampling

- date: `2026-04-27`
- status: `target_only_candidates_sampled`
- git commit: `5feb0e05568c65cc44ff1aadec4973f52b0ccf82`
- model: `Qwen/Qwen3-0.6B`
- eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- samples per example: `8`
- candidate oracle: `14/32`

| Method | Correct | Numeric Coverage |
|---|---:|---:|
| `target_sample_s0` | 6/32 | 32/32 |
| `target_sample_s1` | 7/32 | 32/32 |
| `target_sample_s2` | 7/32 | 32/32 |
| `target_sample_s3` | 10/32 | 32/32 |
| `target_sample_s4` | 5/32 | 32/32 |
| `target_sample_s5` | 9/32 | 32/32 |
| `target_sample_s6` | 4/32 | 32/32 |
| `target_sample_s7` | 7/32 | 32/32 |

## Oracle IDs

`013133cdef4f637c`, `0d2b4f7681973e19`, `14bfbfc94f2c2e7b`, `2f80204fc9e0896f`, `316b525a4fcbfb89`, `3e8a5691f5443495`, `4d780f825bb8541c`, `575d7e83d84c1e67`, `a85be5ec651dac24`, `aee922049c757331`, `de2a795ab37694af`, `e26bdbcab1449bbe`, `e3ab8666238a289e`, `f367349ce0604296`

## Command

```bash
scripts/sample_target_candidate_surface.py --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl --model Qwen/Qwen3-0.6B --samples 8 --temperature 0.9 --top-p 0.95 --seed 43 --device mps --dtype float32 --max-new-tokens 64 --use-chat-template --enable-thinking false --output-jsonl results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl --output-json results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.json --output-md results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.md
```
