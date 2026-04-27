# No-Source Candidate Surface

- date: `2026-04-27`
- status: `no_source_candidate_surface_materialized`
- git commit: `5feb0e05568c65cc44ff1aadec4973f52b0ccf82`
- target set: `results/svamp32_target_sampling_full32_s8_20260427/no_source_surface/source_contrastive_target_set.json`
- target-set SHA256: `647725e7fb079e5c544e1556cab051c124d40982c86b927dd0b639f8a561aaa8`

## Candidate Rows

| Label | Rows | Correct | Expanded | Path | SHA256 |
|---|---:|---:|---|---|---|
| `target_sample_s0` | 32 | 6 | `False` | `results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl` | `650c19fb283d908b1e11af54f19dd0ec8a48e204d38bdd359120d6e4bbd354e5` |
| `target_sample_s1` | 32 | 7 | `False` | `results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl` | `650c19fb283d908b1e11af54f19dd0ec8a48e204d38bdd359120d6e4bbd354e5` |
| `target_sample_s2` | 32 | 7 | `False` | `results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl` | `650c19fb283d908b1e11af54f19dd0ec8a48e204d38bdd359120d6e4bbd354e5` |
| `target_sample_s3` | 32 | 10 | `False` | `results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl` | `650c19fb283d908b1e11af54f19dd0ec8a48e204d38bdd359120d6e4bbd354e5` |
| `target_sample_s4` | 32 | 5 | `False` | `results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl` | `650c19fb283d908b1e11af54f19dd0ec8a48e204d38bdd359120d6e4bbd354e5` |
| `target_sample_s5` | 32 | 9 | `False` | `results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl` | `650c19fb283d908b1e11af54f19dd0ec8a48e204d38bdd359120d6e4bbd354e5` |
| `target_sample_s6` | 32 | 4 | `False` | `results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl` | `650c19fb283d908b1e11af54f19dd0ec8a48e204d38bdd359120d6e4bbd354e5` |
| `target_sample_s7` | 32 | 7 | `False` | `results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl` | `650c19fb283d908b1e11af54f19dd0ec8a48e204d38bdd359120d6e4bbd354e5` |

## Surface Counts

- target correct: `8/32`
- source correct: `6/32`
- clean source-only after no-source baselines: `2`

## Command

```bash
scripts/materialize_no_source_candidate_surface.py --base-target-set results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json --candidate target_sample_s0=path=results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl,method=target_sample_s0 --candidate target_sample_s1=path=results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl,method=target_sample_s1 --candidate target_sample_s2=path=results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl,method=target_sample_s2 --candidate target_sample_s3=path=results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl,method=target_sample_s3 --candidate target_sample_s4=path=results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl,method=target_sample_s4 --candidate target_sample_s5=path=results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl,method=target_sample_s5 --candidate target_sample_s6=path=results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl,method=target_sample_s6 --candidate target_sample_s7=path=results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl,method=target_sample_s7 --min-source-only 0 --date 2026-04-27 --output-dir results/svamp32_target_sampling_full32_s8_20260427/no_source_surface
```
