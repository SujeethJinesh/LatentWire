# SVAMP32 Process Repair Source-Control Gate Manifest

- date: `2026-04-26`
- status: `process_repair_source_stack_fails_gate`
- git commit at run time: `e7cacf4840daade97e2c537628f4fa5b4254c4ea`
- dataset: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- target set: `results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json`
- model: `Qwen/Qwen3-0.6B`
- device: `mps`
- dtype: `float32`
- seed: deterministic greedy generation, no sampling

## Artifacts

| artifact | path | sha256 |
|---|---|---|
| matched process repair JSONL | `results/svamp32_process_repair_source_controls_20260426/matched_process_repair.jsonl` | `a125ac97e1e4c54763739cdcc23e13c8633a8dceb431c3bfbed6c351e5219f6d` |
| matched process repair markdown | `results/svamp32_process_repair_source_controls_20260426/matched_process_repair.md` | `98cf77da5299733d35f5618dd43815b6f90d3dc8904c561835351355b8797dfc` |
| run log | `.debug/svamp32_process_repair_source_controls_20260426/logs/matched_process_repair.log` | `51e0f9286a25fce982dc6fd3de3ae2145bfd495fdc7c6f2018252da597ef1eb4` |
| matched paired input | `.debug/svamp32_process_repair_source_controls_20260426/inputs/matched_target_plus_candidate.jsonl` | `977ddd42bb999ac0e59bf25d687a1c5be997280e976d1b402d6effca1d91704c` |

## Decision

The matched row fails the strict small gate:

- process repair selected route: `10/32`
- target self-repair: `14/32`
- clean residual recovered: `1/6`
- target-self preservation: `1/3`
- selected source candidate rate: `0/32`

Zero-source and shuffled-source repair generations were not run because the
matched row selected target on every example and failed the target-self
comparator before source controls could matter.
