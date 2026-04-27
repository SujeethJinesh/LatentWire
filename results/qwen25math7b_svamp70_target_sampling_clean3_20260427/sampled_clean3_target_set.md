# Extended Target Candidate Set

- date: `2026-04-27`
- status: `target_set_candidate_labels_extended`
- git commit: `6a09be6d62e3804af408cad178b8274b965b7da6`
- base target set: `results/qwen25math7b_qwen3_svamp70_surface_scout_20260427/source_contrastive_target_set.json`
- output target set: `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/sampled_clean3_target_set.json`
- selected IDs: `3`

| Label | Correct | Numeric Coverage | Path |
|---|---:|---:|---|
| `target_sample_s0` | 0/3 | 3/3 | `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s1` | 0/3 | 3/3 | `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s2` | 0/3 | 3/3 | `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s3` | 1/3 | 3/3 | `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s4` | 0/3 | 3/3 | `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s5` | 0/3 | 3/3 | `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s6` | 0/3 | 3/3 | `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s7` | 0/3 | 3/3 | `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s8` | 0/3 | 3/3 | `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s9` | 0/3 | 3/3 | `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s10` | 0/3 | 3/3 | `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s11` | 0/3 | 3/3 | `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s12` | 0/3 | 3/3 | `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s13` | 0/3 | 3/3 | `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s14` | 0/3 | 3/3 | `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s15` | 0/3 | 3/3 | `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl` |

## Command

```bash
scripts/extend_target_set_candidate_labels.py --base-target-set results/qwen25math7b_qwen3_svamp70_surface_scout_20260427/source_contrastive_target_set.json --id-fields clean_source_only --candidate target_sample_s0=path=results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s0 --candidate target_sample_s1=path=results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s1 --candidate target_sample_s2=path=results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s2 --candidate target_sample_s3=path=results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s3 --candidate target_sample_s4=path=results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s4 --candidate target_sample_s5=path=results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s5 --candidate target_sample_s6=path=results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s6 --candidate target_sample_s7=path=results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s7 --candidate target_sample_s8=path=results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s8 --candidate target_sample_s9=path=results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s9 --candidate target_sample_s10=path=results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s10 --candidate target_sample_s11=path=results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s11 --candidate target_sample_s12=path=results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s12 --candidate target_sample_s13=path=results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s13 --candidate target_sample_s14=path=results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s14 --candidate target_sample_s15=path=results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s15 --date 2026-04-27 --output-json results/qwen25math7b_svamp70_target_sampling_clean3_20260427/sampled_clean3_target_set.json --output-md results/qwen25math7b_svamp70_target_sampling_clean3_20260427/sampled_clean3_target_set.md --manifest-json results/qwen25math7b_svamp70_target_sampling_clean3_20260427/sampled_clean3_target_set_manifest.json
```
