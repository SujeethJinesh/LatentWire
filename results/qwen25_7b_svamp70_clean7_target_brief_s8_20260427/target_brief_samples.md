# Candidate Surface Sampling

- date: `2026-04-27`
- status: `candidate_surface_sampled`
- git commit: `f17c315e2170b872fcfed6402f94f2d39b03a01a`
- model: `Qwen/Qwen3-0.6B`
- eval file: `results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/clean7_eval.jsonl`
- samples per example: `8`
- candidate oracle: `4/7`

| Method | Correct | Numeric Coverage |
|---|---:|---:|
| `target_brief_sample_s0` | 2/7 | 7/7 |
| `target_brief_sample_s1` | 2/7 | 7/7 |
| `target_brief_sample_s2` | 4/7 | 7/7 |
| `target_brief_sample_s3` | 3/7 | 7/7 |
| `target_brief_sample_s4` | 3/7 | 7/7 |
| `target_brief_sample_s5` | 3/7 | 7/7 |
| `target_brief_sample_s6` | 2/7 | 7/7 |
| `target_brief_sample_s7` | 3/7 | 7/7 |

## Oracle IDs

`3c5aeb08941dbb6d`, `ce08a3a269bf0151`, `de1bf4d142544e5b`, `e099e405e8d1a66b`

## Command

```bash
scripts/sample_target_candidate_surface.py --eval-file results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/clean7_eval.jsonl --model Qwen/Qwen3-0.6B --samples 8 --method-prefix target_brief_sample --temperature 0.9 --top-p 0.95 --seed 71 --device mps --dtype float32 --max-new-tokens 64 --prompt-mode source_reasoning --source-reasoning-mode brief_analysis --use-chat-template --enable-thinking false --output-jsonl results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/target_brief_samples.jsonl --output-json results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/target_brief_samples.json --output-md results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/target_brief_samples.md
```
