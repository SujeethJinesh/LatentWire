# Extended Target Candidate Set

- date: `2026-04-27`
- status: `target_set_candidate_labels_extended`
- git commit: `afbf022b1e7e1e96c1bd76f72c28e6f538f73abf`
- base target set: `results/no_source_candidate_surface_20260427/source_contrastive_target_set.json`
- output target set: `results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json`
- selected IDs: `3`

| Label | Correct | Numeric Coverage | Path |
|---|---:|---:|---|
| `target_sample_s0` | 0/3 | 3/3 | `results/target_only_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s1` | 1/3 | 3/3 | `results/target_only_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s2` | 1/3 | 3/3 | `results/target_only_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s3` | 0/3 | 3/3 | `results/target_only_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s4` | 1/3 | 3/3 | `results/target_only_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s5` | 0/3 | 3/3 | `results/target_only_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s6` | 0/3 | 3/3 | `results/target_only_sampling_clean3_20260427/target_only_samples.jsonl` |
| `target_sample_s7` | 0/3 | 3/3 | `results/target_only_sampling_clean3_20260427/target_only_samples.jsonl` |

## Command

```bash
scripts/extend_target_set_candidate_labels.py --base-target-set results/no_source_candidate_surface_20260427/source_contrastive_target_set.json --id-fields clean_source_only --candidate target_sample_s0=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s0 --candidate target_sample_s1=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s1 --candidate target_sample_s2=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s2 --candidate target_sample_s3=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s3 --candidate target_sample_s4=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s4 --candidate target_sample_s5=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s5 --candidate target_sample_s6=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s6 --candidate target_sample_s7=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s7 --date 2026-04-27 --output-json results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json --output-md results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.md --manifest-json results/target_only_sampling_clean3_20260427/sampled_clean3_target_set_manifest.json
```
