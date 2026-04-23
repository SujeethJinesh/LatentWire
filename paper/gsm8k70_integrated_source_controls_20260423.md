# GSM8K70 Integrated Source-Control Gate

Date: 2026-04-23

## Status

This is a source-control and seed-repeat gate for the existing
`dynalign_module_replace_residrank16` lane. It is not a new method result.

The seed-0 row remains alive as a source-dependent same-family mechanism clue.
The branch is still blocked as a paper method because the finite repeat does
not beat target.

## Commands

Seed 0:

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 70 \
  --eval-file data/gsm8k_eval_70.jsonl \
  --materialized-eval-file results/gsm8k70_seed_repeat_full_20260422/_artifacts/gsm8k_eval_70.jsonl \
  --baseline-results-dir results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke \
  --results-dir .debug/gsm8k70_integrated_source_controls_20260423/seed0 \
  --checkpoints-dir checkpoints/gsm8k_contract_residual_sweep_20260421 \
  --seed 0 \
  --gate 0.10 \
  --run-source-controls \
  --source-control-random-salt 0
```

The seed-0 live row was regenerated in the wrapper. The zero-source and
shuffled-source prediction files were reused from the prior matched
source-control run via `.debug` row-local symlinks, then re-analyzed by the new
wrapper path.

Seed 3:

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 70 \
  --eval-file data/gsm8k_eval_70.jsonl \
  --materialized-eval-file results/gsm8k70_seed_repeat_full_20260422/_artifacts/gsm8k_eval_70.jsonl \
  --baseline-results-dir results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke \
  --results-dir .debug/gsm8k70_integrated_source_controls_20260423/seed3 \
  --checkpoints-dir checkpoints/gsm8k_contract_residual_sweep_20260421 \
  --seed 3 \
  --gate 0.10 \
  --run-source-controls \
  --source-control-random-salt 3
```

The seed-3 live prediction file was reused from the prior finite repeat because
it was already on the same frozen slice/checkpoint contract. The wrapper
recomputed the row checks and correctly skipped source controls because the
live gate failed.

## Readout

| Seed | Status | Accuracy | Correct | Pair vs target | Numeric coverage | Source-control status |
|---:|---|---:|---:|---:|---:|---|
| 0 | `ok` | `0.1143` | `8/70` | `6/2/62` | `70/70` | `source_controls_support_matched_source_signal` |
| 3 | `ok` | `0.0286` | `2/70` | `1/3/66` | `69/70` | `not_run_live_gate_failed` |

Seed-0 controls with target fallback:

| Control | Correct | Pair vs target | Pair vs live | Live-win retention | Coverage | Deranged |
|---|---:|---:|---:|---:|---:|---:|
| `zero_source` | `4/70` | `0/0/70` | `2/6/62` | `0/6` | `70/70` | `False` |
| `shuffled_source_salt0` | `4/70` | `0/0/70` | `2/6/62` | `0/6` | `70/70` | `True` |

Artifact hashes:

```text
0ada9e55c0c3518b36049c2a99f817f0f62a6b21e5be05f21665896972851d3f  .debug/gsm8k70_integrated_source_controls_20260423/seed0/gsm8k_contract_residual_sweep_20260421.json
c6bf310dea326ddbee116e656c64db5f5556b8fca163818fbb29e44ad630ed86  .debug/gsm8k70_integrated_source_controls_20260423/seed0/dynalign_module_replace_residrank16/source_controls/source_control_readout.json
f2762cb777a71ef220df36c561dca964ce893800e1eb3bd9dc34f60283592a51  .debug/gsm8k70_integrated_source_controls_20260423/seed3/gsm8k_contract_residual_sweep_20260421.json
```

## Interpretation

The seed-0 result strengthens the source-dependence interpretation: the live
candidate has `6` wins over target and both zero-source and shuffled-source
controls retain `0/6` of those wins while preserving full numeric coverage
under target fallback.

The method branch still cannot be promoted. Seed 3 is finite and ordered-ID
clean, but it is target-negative with `1` win and `3` losses. The blocker is
therefore robustness/selection, not source-dependence on seed 0.

## Subagent Inputs

- Literature/method sidecar recommended a control-calibrated
  innovation accept/fallback rule: accept the latent intervention only when a
  predeclared source-innovation score clears a threshold calibrated so
  zero/shuffled controls almost never trigger.
- Benchmark sidecar confirmed seed 0 is the right controlled rerun and seed 3
  should be expected to fail before controls.
- Repo-audit sidecar identified positional row selection in
  `scripts/build_gsm8k_contract_manifest.py` as the highest-value cleanup
  before any new artifact manifest refresh.

## Verification

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_build_gsm8k_contract_manifest.py \
  tests/test_gsm8k_contract_residual_sweep.py \
  tests/test_analyze_gsm8k_source_controls.py
```

Result: `41 passed in 0.10s`.

## Decision

Alive:

- Seed-0 matched-source same-family signal.
- Source-control-aware accept/fallback as the next method idea.

Weakened:

- Raw `dynalign_module_replace_residrank16` as a method, because seed 3 fails
  the live gate and seeds 1/2 remain nonfinite.

Saturated:

- Further uncontrolled dynalign sweeps on this lane. More raw rows are unlikely
  to solve reviewer concerns without a target-safe selection mechanism.

Next exact gate:

Implement an offline control-calibrated accept/fallback replay over the seed-0
and seed-3 artifacts. The decisive question is whether a predeclared score can
retain a useful subset of seed-0 wins while accepting zero/shuffled controls
rarely and avoiding seed-3 harms. If it collapses to target or accepts the
wrong examples, demote dynalign to a brittle mechanism probe and move the main
method effort back to learned connector/conditional innovation designs.
