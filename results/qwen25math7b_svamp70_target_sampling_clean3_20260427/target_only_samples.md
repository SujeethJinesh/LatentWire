# Target-Only Candidate Sampling

- date: `2026-04-27`
- status: `target_only_candidates_sampled`
- git commit: `6a09be6d62e3804af408cad178b8274b965b7da6`
- model: `Qwen/Qwen3-0.6B`
- eval file: `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/clean_source_only_eval.jsonl`
- samples per example: `16`
- candidate oracle: `1/3`

| Method | Correct | Numeric Coverage |
|---|---:|---:|
| `target_sample_s0` | 0/3 | 3/3 |
| `target_sample_s1` | 0/3 | 3/3 |
| `target_sample_s10` | 0/3 | 3/3 |
| `target_sample_s11` | 0/3 | 3/3 |
| `target_sample_s12` | 0/3 | 3/3 |
| `target_sample_s13` | 0/3 | 3/3 |
| `target_sample_s14` | 0/3 | 3/3 |
| `target_sample_s15` | 0/3 | 3/3 |
| `target_sample_s2` | 0/3 | 3/3 |
| `target_sample_s3` | 1/3 | 3/3 |
| `target_sample_s4` | 0/3 | 3/3 |
| `target_sample_s5` | 0/3 | 3/3 |
| `target_sample_s6` | 0/3 | 3/3 |
| `target_sample_s7` | 0/3 | 3/3 |
| `target_sample_s8` | 0/3 | 3/3 |
| `target_sample_s9` | 0/3 | 3/3 |

## Oracle IDs

`14bfbfc94f2c2e7b`

## Command

```bash
scripts/sample_target_candidate_surface.py --eval-file results/qwen25math7b_svamp70_target_sampling_clean3_20260427/clean_source_only_eval.jsonl --model Qwen/Qwen3-0.6B --samples 16 --temperature 0.9 --top-p 0.95 --seed 17 --device mps --dtype float32 --max-new-tokens 64 --use-chat-template --enable-thinking false --output-jsonl results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl --output-json results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.json --output-md results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.md
```
