# SVAMP32 Target-Safe Oracle Replay

- date: `2026-04-26`
- status: `target_safe_dynalign_selector_branch_killed`
- scale-up rung: strict small exact-ID gate
- current ICLR readiness: not ready
- estimated distance: one stable deployable positive method plus larger-slice,
  seed-repeat, source-control, systems, and cross-family gates

## Start Status

- current story: C2C and target-side self-repair expose real headroom on frozen
  SVAMP32, but previous source-readout and selector/repair branches fail clean
  source-necessary controls.
- blocker: determine whether existing target-safe dynalign/query-pool rows have
  enough clean residual signal to justify another output-aware selector or
  repair run.
- live branch entering gate: target-safe output-aware dynalign selector/repair.
- highest-priority gate: recover at least `2/6` clean source-necessary C2C-only
  IDs while preserving target-self repair.

## What Changed

Added `scripts/analyze_svamp32_target_safe_oracle.py`, a strict replay analyzer
that computes:

- the best target-safe oracle over a baseline fallback and candidate rows
- the matching target-safe oracle over source-destroying control rows
- clean source-necessary residual IDs after subtracting the control oracle

This is an upper-bound test for selector/repair work. If the oracle over
existing matched candidates cannot clear the clean gate, another learned
selector over the same candidates is not worth running.

## Command

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_target_safe_oracle.py \
  --target target=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate \
  --baseline target_self_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/target_self_repair_exact32.jsonl,method=target_self_repair \
  --candidate dynalign_salt1=path=results/process_repair_holdout_20260421/qwen_svamp70_dynalign_prefdist_asym_kv_random_r025_v075_cal16_chat_salt1_telemetry.jsonl,method=rotalign_kv \
  --candidate dynalign_salt2=path=results/process_repair_holdout_20260421/qwen_svamp70_dynalign_prefdist_asym_kv_random_r025_v075_cal16_chat_salt2_telemetry.jsonl,method=rotalign_kv \
  --candidate query_pool=path=results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_matched.jsonl,method=rotalign_kv \
  --control dynalign_salt1_zero=path=results/svamp32_dynalign_source_controls_20260423/dynalign_salt1_zero_source.jsonl,method=rotalign_kv \
  --control dynalign_salt1_shuffle=path=results/svamp32_dynalign_source_controls_20260423/dynalign_salt1_shuffled_source_salt1.jsonl,method=rotalign_kv \
  --control dynalign_salt2_zero=path=results/svamp32_dynalign_source_controls_20260423/dynalign_salt2_zero_source.jsonl,method=rotalign_kv \
  --control dynalign_salt2_shuffle=path=results/svamp32_dynalign_source_controls_20260423/dynalign_salt2_shuffled_source_salt2.jsonl,method=rotalign_kv \
  --control query_pool_zero=path=results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_zero_source.jsonl,method=rotalign_kv \
  --control query_pool_shuffle=path=results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_shuffled_source_salt1.jsonl,method=rotalign_kv \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --min-clean-source-necessary 2 \
  --max-losses-vs-target 1 \
  --output-json results/svamp32_target_safe_oracle_replay_20260426/oracle.json \
  --output-md results/svamp32_target_safe_oracle_replay_20260426/oracle.md
```

## Result

| row | correct | C2C-only | clean residual | target losses |
|---|---:|---:|---:|---:|
| target_self_repair | `14/32` | `3` | `0/6` | `0` |
| dynalign_salt1 | `9/32` | `1` | `0/6` | `2` |
| dynalign_salt2 | `8/32` | `2` | `1/6` | `4` |
| query_pool | `9/32` | `1` | `0/6` | `1` |
| target-safe candidate oracle | `18/32` | `5` | `1/6` | `0` |
| target-safe control oracle | `18/32` | `5` | `1/6` | `0` |

Clean accounting:

- candidate-oracle clean ID: `e3ab8666238a289e`
- control-oracle clean ID: `aee922049c757331`
- clean source-necessary count after subtracting controls: `1/6`
- required count: `2/6`

## Decision

Kill target-safe output-aware dynalign selector/repair over these existing
candidates. Even an oracle selector over target_self_repair, dynalign salts,
and query-pool transport cannot clear the clean source-necessary gate. The
control oracle also reaches the same overall score and recovers a clean C2C
residual ID, so another selector would mostly select among artifacts already
available to source-destroying controls.

Promoted next branch: a genuinely learned communication protocol, not a replay
selector over saturated candidates. The highest-value branch is a minimal
target-conditioned soft-token / Q-Former-style connector with explicit
source-destroying controls, trained against the C2C-over-target_self residual
surface before any larger scale-up.

## Artifacts

- replay JSON:
  - `results/svamp32_target_safe_oracle_replay_20260426/oracle.json`
  - sha256: `1cb42394749c7bd1b80439bccc65441b63aba3b59ad9733994f080c337400746`
- replay markdown:
  - `results/svamp32_target_safe_oracle_replay_20260426/oracle.md`
  - sha256: `06f2126b891af93c102c7482b9cacf3e39154ff0c583e8dd6e814ff3c2d638b2`

## Tests

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_target_safe_oracle.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_target_safe_oracle.py`

## Next Exact Gate

Implement or locate the smallest target-conditioned soft-token / learned-query
connector that can train on frozen source and target traces, then run it on the
SVAMP32 clean residual target set with matched, zero-source, shuffled-source,
target-only, and slots-only controls. Promote only if it recovers at least
`2/6` clean source-necessary IDs with no target-self regression.
