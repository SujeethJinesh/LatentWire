# No-Source Candidate Surface

- date: `2026-04-27`
- status: `no_source_candidate_surface_materialized`
- git commit: `76abab281d43ec30e509d58e750c3232f2adcb0e`
- target set: `results/no_source_candidate_surface_20260427/source_contrastive_target_set.json`
- target-set SHA256: `fb615786f89643c6208909534f59896c2f7d8987b29043842941a769e52e26aa`

## Candidate Rows

| Label | Rows | Correct | Expanded | Path | SHA256 |
|---|---:|---:|---|---|---|
| `target_self_repair` | 70 | 35 | `False` | `results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_process_repair_controls_telemetry.jsonl` | `3d268956c22d8d88e6107535cb59a2013ce06769c5f08f92e6c07b73fd345596` |
| `selected_route_no_repair` | 70 | 22 | `False` | `results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_process_repair_controls_telemetry.jsonl` | `3d268956c22d8d88e6107535cb59a2013ce06769c5f08f92e6c07b73fd345596` |
| `process_repair` | 70 | 35 | `False` | `results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_process_repair_controls_telemetry.jsonl` | `3d268956c22d8d88e6107535cb59a2013ce06769c5f08f92e6c07b73fd345596` |
| `zero_source_pool_seed_0` | 70 | 23 | `True` | `results/no_source_candidate_surface_20260427/zero_source_pool_seed_0.jsonl` | `b9cd888e95303db3cf1a36120b70c984ff8196f08f29a08e54b576afcf33baf3` |
| `zero_source_pool_seed_1` | 70 | 17 | `True` | `results/no_source_candidate_surface_20260427/zero_source_pool_seed_1.jsonl` | `077dfa9f4a2c1b13a2895e14e0335eb51db36c9457e050cf13a8bfd049671137` |
| `zero_source_pool_seed_2` | 70 | 16 | `True` | `results/no_source_candidate_surface_20260427/zero_source_pool_seed_2.jsonl` | `7e1436b427297deaa2086e62ea747191c89df698f3b7a0d13d39b9bdc0b6a535` |

## Surface Counts

- target correct: `21/70`
- source correct: `13/70`
- clean source-only after no-source baselines: `3`

## Command

```bash
scripts/materialize_no_source_candidate_surface.py --base-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json --candidate target_self_repair=path=results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_process_repair_controls_telemetry.jsonl,method=target_self_repair --candidate selected_route_no_repair=path=results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_process_repair_controls_telemetry.jsonl,method=selected_route_no_repair --candidate process_repair=path=results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_process_repair_controls_telemetry.jsonl,method=process_repair_selected_route --expand-candidate-scores zero_source_pool=path=results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_process_repair_controls_telemetry.jsonl,method=selected_route_no_repair --min-source-only 0 --date 2026-04-27 --output-dir results/no_source_candidate_surface_20260427
```
