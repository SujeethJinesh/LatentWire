# Candidate Surface Sampling

- date: `2026-04-27`
- status: `candidate_surface_sampled`
- git commit: `ef4886ed050f7ee21c8746a42e673a86c5fc1fe1`
- model: `Qwen/Qwen3-0.6B`
- eval file: `results/svamp32_source_sampling_newclean2_s16_20260427/newclean2_eval.jsonl`
- samples per example: `16`
- candidate oracle: `0/2`

| Method | Correct | Numeric Coverage |
|---|---:|---:|
| `target_direct_sample_s0` | 0/2 | 2/2 |
| `target_direct_sample_s1` | 0/2 | 2/2 |
| `target_direct_sample_s10` | 0/2 | 2/2 |
| `target_direct_sample_s11` | 0/2 | 2/2 |
| `target_direct_sample_s12` | 0/2 | 2/2 |
| `target_direct_sample_s13` | 0/2 | 2/2 |
| `target_direct_sample_s14` | 0/2 | 2/2 |
| `target_direct_sample_s15` | 0/2 | 2/2 |
| `target_direct_sample_s2` | 0/2 | 2/2 |
| `target_direct_sample_s3` | 0/2 | 2/2 |
| `target_direct_sample_s4` | 0/2 | 2/2 |
| `target_direct_sample_s5` | 0/2 | 2/2 |
| `target_direct_sample_s6` | 0/2 | 2/2 |
| `target_direct_sample_s7` | 0/2 | 2/2 |
| `target_direct_sample_s8` | 0/2 | 2/2 |
| `target_direct_sample_s9` | 0/2 | 2/2 |

## Oracle IDs

none

## Command

```bash
scripts/sample_target_candidate_surface.py --eval-file results/svamp32_source_sampling_newclean2_s16_20260427/newclean2_eval.jsonl --model Qwen/Qwen3-0.6B --samples 16 --method-prefix target_direct_sample --temperature 0.9 --top-p 0.95 --seed 171 --device mps --dtype float32 --max-new-tokens 64 --prompt-mode direct --use-chat-template --enable-thinking false --output-jsonl results/svamp32_source_sampling_newclean2_s16_20260427/target_direct_samples.jsonl --output-json results/svamp32_source_sampling_newclean2_s16_20260427/target_direct_samples.json --output-md results/svamp32_source_sampling_newclean2_s16_20260427/target_direct_samples.md
```
