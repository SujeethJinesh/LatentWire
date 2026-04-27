# Extended Target Candidate Set

- date: `2026-04-27`
- status: `target_set_candidate_labels_extended`
- git commit: `614f2a2d5482b3d5007b005b4f7df86c23c5479d`
- base target set: `results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json`
- output target set: `results/svamp32_target_sampling_clean6_20260427/sampled_clean6_target_set.json`
- selected IDs: `6`

| Label | Correct | Numeric Coverage | Path |
|---|---:|---:|---|
| `target_sample_s0` | 0/6 | 6/6 | `results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl` |
| `target_sample_s1` | 0/6 | 6/6 | `results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl` |
| `target_sample_s2` | 0/6 | 6/6 | `results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl` |
| `target_sample_s3` | 1/6 | 6/6 | `results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl` |
| `target_sample_s4` | 1/6 | 6/6 | `results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl` |
| `target_sample_s5` | 1/6 | 6/6 | `results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl` |
| `target_sample_s6` | 1/6 | 6/6 | `results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl` |
| `target_sample_s7` | 1/6 | 6/6 | `results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl` |
| `target_sample_s8` | 1/6 | 6/6 | `results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl` |
| `target_sample_s9` | 1/6 | 6/6 | `results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl` |
| `target_sample_s10` | 1/6 | 6/6 | `results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl` |
| `target_sample_s11` | 0/6 | 6/6 | `results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl` |
| `target_sample_s12` | 0/6 | 6/6 | `results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl` |
| `target_sample_s13` | 1/6 | 6/6 | `results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl` |
| `target_sample_s14` | 1/6 | 6/6 | `results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl` |
| `target_sample_s15` | 1/6 | 6/6 | `results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl` |

## Command

```bash
scripts/extend_target_set_candidate_labels.py --base-target-set results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json --ids 1d50b408c8f5cd2c 3e8a5691f5443495 47464cc0b064f172 575d7e83d84c1e67 6e9745b37ab6fc45 de1bf4d142544e5b --override-clean-residual-ids 1d50b408c8f5cd2c 3e8a5691f5443495 47464cc0b064f172 575d7e83d84c1e67 6e9745b37ab6fc45 de1bf4d142544e5b --candidate target_sample_s0=path=results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl,method=target_sample_s0 --candidate target_sample_s1=path=results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl,method=target_sample_s1 --candidate target_sample_s2=path=results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl,method=target_sample_s2 --candidate target_sample_s3=path=results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl,method=target_sample_s3 --candidate target_sample_s4=path=results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl,method=target_sample_s4 --candidate target_sample_s5=path=results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl,method=target_sample_s5 --candidate target_sample_s6=path=results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl,method=target_sample_s6 --candidate target_sample_s7=path=results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl,method=target_sample_s7 --candidate target_sample_s8=path=results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl,method=target_sample_s8 --candidate target_sample_s9=path=results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl,method=target_sample_s9 --candidate target_sample_s10=path=results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl,method=target_sample_s10 --candidate target_sample_s11=path=results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl,method=target_sample_s11 --candidate target_sample_s12=path=results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl,method=target_sample_s12 --candidate target_sample_s13=path=results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl,method=target_sample_s13 --candidate target_sample_s14=path=results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl,method=target_sample_s14 --candidate target_sample_s15=path=results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl,method=target_sample_s15 --date 2026-04-27 --output-json results/svamp32_target_sampling_clean6_20260427/sampled_clean6_target_set.json --output-md results/svamp32_target_sampling_clean6_20260427/sampled_clean6_target_set.md --manifest-json results/svamp32_target_sampling_clean6_20260427/extend_manifest.json
```
