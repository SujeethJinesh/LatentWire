# Candidate Surface Sampling

- date: `2026-04-27`
- status: `candidate_surface_sampled`
- git commit: `f17c315e2170b872fcfed6402f94f2d39b03a01a`
- model: `Qwen/Qwen2.5-Math-1.5B`
- eval file: `results/gsm_source_residual_prompt_control_clean2_20260427/gsm_clean2_eval.jsonl`
- samples per example: `8`
- candidate oracle: `1/2`

| Method | Correct | Numeric Coverage |
|---|---:|---:|
| `source_brief_sample_s0` | 0/2 | 2/2 |
| `source_brief_sample_s1` | 1/2 | 2/2 |
| `source_brief_sample_s2` | 0/2 | 0/2 |
| `source_brief_sample_s3` | 0/2 | 1/2 |
| `source_brief_sample_s4` | 0/2 | 2/2 |
| `source_brief_sample_s5` | 0/2 | 1/2 |
| `source_brief_sample_s6` | 0/2 | 2/2 |
| `source_brief_sample_s7` | 0/2 | 2/2 |

## Oracle IDs

`1deed634dcd7d229`

## Command

```bash
scripts/sample_target_candidate_surface.py --eval-file results/gsm_source_residual_prompt_control_clean2_20260427/gsm_clean2_eval.jsonl --model Qwen/Qwen2.5-Math-1.5B --samples 8 --method-prefix source_brief_sample --temperature 0.9 --top-p 0.95 --seed 211 --device mps --dtype float32 --max-new-tokens 64 --prompt-mode source_reasoning --source-reasoning-mode brief_analysis --use-chat-template --enable-thinking false --output-jsonl results/gsm_source_residual_prompt_control_clean2_20260427/source_brief_samples.jsonl --output-json results/gsm_source_residual_prompt_control_clean2_20260427/source_brief_samples.json --output-md results/gsm_source_residual_prompt_control_clean2_20260427/source_brief_samples.md
```
