# Candidate Surface Sampling

- date: `2026-04-27`
- status: `candidate_surface_sampled`
- git commit: `ef4886ed050f7ee21c8746a42e673a86c5fc1fe1`
- model: `Qwen/Qwen3-0.6B`
- eval file: `results/svamp32_source_sampling_newclean2_s16_20260427/newclean2_eval.jsonl`
- samples per example: `16`
- candidate oracle: `2/2`

| Method | Correct | Numeric Coverage |
|---|---:|---:|
| `target_brief_sample_s0` | 1/2 | 2/2 |
| `target_brief_sample_s1` | 2/2 | 2/2 |
| `target_brief_sample_s10` | 1/2 | 2/2 |
| `target_brief_sample_s11` | 2/2 | 2/2 |
| `target_brief_sample_s12` | 1/2 | 2/2 |
| `target_brief_sample_s13` | 2/2 | 2/2 |
| `target_brief_sample_s14` | 1/2 | 2/2 |
| `target_brief_sample_s15` | 1/2 | 2/2 |
| `target_brief_sample_s2` | 2/2 | 2/2 |
| `target_brief_sample_s3` | 1/2 | 2/2 |
| `target_brief_sample_s4` | 1/2 | 2/2 |
| `target_brief_sample_s5` | 1/2 | 2/2 |
| `target_brief_sample_s6` | 2/2 | 2/2 |
| `target_brief_sample_s7` | 2/2 | 2/2 |
| `target_brief_sample_s8` | 2/2 | 2/2 |
| `target_brief_sample_s9` | 1/2 | 2/2 |

## Oracle IDs

`6e9745b37ab6fc45`, `de1bf4d142544e5b`

## Command

```bash
scripts/sample_target_candidate_surface.py --eval-file results/svamp32_source_sampling_newclean2_s16_20260427/newclean2_eval.jsonl --model Qwen/Qwen3-0.6B --samples 16 --method-prefix target_brief_sample --temperature 0.9 --top-p 0.95 --seed 171 --device mps --dtype float32 --max-new-tokens 64 --prompt-mode source_reasoning --source-reasoning-mode brief_analysis --use-chat-template --enable-thinking false --output-jsonl results/svamp32_source_sampling_newclean2_s16_20260427/target_brief_samples.jsonl --output-json results/svamp32_source_sampling_newclean2_s16_20260427/target_brief_samples.json --output-md results/svamp32_source_sampling_newclean2_s16_20260427/target_brief_samples.md
```
