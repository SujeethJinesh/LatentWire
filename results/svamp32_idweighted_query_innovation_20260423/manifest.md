# SVAMP32 ID-Weighted Query Innovation Manifest

- date: `2026-04-23`
- branch: `id-weighted query innovation`
- checkpoint: `.debug/checkpoints_svamp32_conditional_innovation_20260423/id_weighted_query_innovation/qwen25_to_qwen3_svamp32_idweighted_query_innovation_r16_bank16_seed1.pt`
- strict target-set gate: `no_candidate_passes_target_self_repair_gate`
- target-self sidecar oracle bound: `oracle_sidecar_bound_fails_gate`

## Main Evidence

- target-alone: `8/32`
- C2C teacher: `16/32`
- target_self_repair: `14/32`
- best matched candidate: `gate015`, `10/32`
- best matched clean residual recovered: `1/6`
- best matched clean source-necessary recovered: `1/6`
- oracle target_self_repair + clean source sidecar bound: `15/32`
- sidecar-bound failing criteria: `min_correct`, `min_clean_source_necessary`

## Artifact Hashes

- `c2c_teacher_probe_gate015_targetself_translated_zero.json`
  - sha256: `d9bd77820e0afec15cc088c2b67fe8afb37305a574ad955866e3ae9f5001cd87`
- `c2c_teacher_probe_gate015_targetself_translated_zero.md`
  - sha256: `c11056f29db3151b7572c6d7fbd0bd1e99436b8c2d202a638225004de4b7798a`
- `paper_gate_gate015_targetself_translated_zero.json`
  - sha256: `6443162d9a8fb7b900e4a0f131dbeb5ecfb7f54ef19b58bd1beee77f90372255`
- `paper_gate_gate015_targetself_translated_zero.md`
  - sha256: `b4e2bdbfc245a06ec145b63fdc88b822e6859d509789e00a7f55750b0c8b5c9c`
- `source_sidecar_bound_gate015_targetself_translated_zero.json`
  - sha256: `678015b227017b2c679d2708ff89311fa749407814320c138be49789aeb3ad08`
- `source_sidecar_bound_gate015_targetself_translated_zero.md`
  - sha256: `2b4958c03166acd9e78b55e9c3f9a65647c8136606093acf9a94f14e06c87ed8`
