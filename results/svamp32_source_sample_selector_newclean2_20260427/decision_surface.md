# Candidate Pool Decision Surface

- date: `2026-04-27`
- status: `candidate_pool_decision_surface_ready`
- git commit: `ef4886ed050f7ee21c8746a42e673a86c5fc1fe1`
- base target set: `results/svamp32_target_sampling_full32_s8_20260427/no_source_surface/source_contrastive_target_set.json`
- reference rows: `32`
- clean decision IDs: `2`

## Clean IDs

- `6e9745b37ab6fc45`
- `de1bf4d142544e5b`

## Extra Candidate Labels

- `source_sample_s0` from `results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl` method `target_sample_s0`
- `source_sample_s1` from `results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl` method `target_sample_s1`
- `source_sample_s2` from `results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl` method `target_sample_s2`
- `source_sample_s3` from `results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl` method `target_sample_s3`

## Command

```bash
scripts/build_candidate_pool_decision_surface.py --base-target-set results/svamp32_target_sampling_full32_s8_20260427/no_source_surface/source_contrastive_target_set.json --extra-candidate label=source_sample_s0,path=results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl,method=target_sample_s0 --extra-candidate label=source_sample_s1,path=results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl,method=target_sample_s1 --extra-candidate label=source_sample_s2,path=results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl,method=target_sample_s2 --extra-candidate label=source_sample_s3,path=results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl,method=target_sample_s3 --clean-id 6e9745b37ab6fc45 --clean-id de1bf4d142544e5b --date 2026-04-27 --output-json results/svamp32_source_sample_selector_newclean2_20260427/decision_surface.json --output-md results/svamp32_source_sample_selector_newclean2_20260427/decision_surface.md
```
