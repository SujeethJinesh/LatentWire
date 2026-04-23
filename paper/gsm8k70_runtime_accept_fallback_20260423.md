# GSM8K70 Runtime Accept/Fallback Gate

Date: 2026-04-23

## Status

This is the runtime validation of the frozen
`selector_gap_min_ge_q0p7_numeric_changed` accept/fallback policy from
`paper/gsm8k70_accept_fallback_replay_20260423.md`.

The result kills this selector-gap policy as a publishable method branch. The
seed-0 live row reproduces the offline positive, but fresh zero-source and
shuffled-source controls also pass the same frozen selector and retain most of
the apparent live wins.

## Command

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
  --results-dir .debug/gsm8k70_runtime_accept_fallback_20260423/seed0 \
  --checkpoints-dir checkpoints/gsm8k_contract_residual_sweep_20260421 \
  --seed 0 \
  --gate 0.10 \
  --run-source-controls \
  --source-control-random-salt 0 \
  --accept-fallback-score-field selector_gap_min \
  --accept-fallback-threshold 0.029237359762191772
```

The run used fresh raw control generations. The seed-3 repeat was not run
because seed 0 already failed the source-control gate.

## Artifact Hashes

```text
1af3ea2d9abcfe01db7da65c4b9b05ee19d46c242e6dd96310e96ea874f02ba3  .debug/gsm8k70_runtime_accept_fallback_20260423/seed0/gsm8k_contract_residual_sweep_20260421.json
99e81ff382dfec5e6d6b74a7d99d1a5eb5230af0b70953fa2887e3cc4b8c168f  .debug/gsm8k70_runtime_accept_fallback_20260423/seed0/gsm8k_contract_residual_sweep_20260421.md
8bd30ce4b99de871c54b847b37eb0a7c0d0ec7f5af2e398624de9d9aee9de09f  .debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16_accept_selector_gap_min_ge_0p02923736/source_controls/source_control_readout.json
9798ca3466759d997a590857ab8de1200405a7082678a43d1b282d9bfd708bd7  .debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16_accept_selector_gap_min_ge_0p02923736/source_controls/source_control_readout.md
6d1224fb76fcddd39c7c06638f0f1435cf29cd9ed514e5482fb91ed35bcbbad1  .debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16_accept_selector_gap_min_ge_0p02923736.jsonl
057293a05d1d09897049e6dbbeb50fc2b80e2a9b4fd8628e35b065d81f665735  .debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16_accept_selector_gap_min_ge_0p02923736/source_controls/zero_source_accept_selector_gap_min_ge_0p02923736.jsonl
2b893d2ea6cc87caf5f19d3c7f677d5a6da3883d091d2ae23ec9a72fa6a75758  .debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16_accept_selector_gap_min_ge_0p02923736/source_controls/shuffled_source_salt0_accept_selector_gap_min_ge_0p02923736.jsonl
```

## Readout

Target baseline: `4/70`.

| Row | Correct | Pair vs target | Accepted | Accepted correct IDs |
|---|---:|---:|---:|---|
| live matched source | `7/70` | `3/0/67` | `13` | `645a38303f97c7b7`, `e100c479d9fc22f8`, `31715a2b361f0b6d` |
| zero source | `6/70` | `2/0/68` | `14` | `e100c479d9fc22f8`, `31715a2b361f0b6d` |
| shuffled source | `6/70` | `2/0/68` | `14` | `e100c479d9fc22f8`, `31715a2b361f0b6d` |

Source-control gate: `source_controls_do_not_clear_gate`.

Both controls retain `2/3` live wins:

```text
31715a2b361f0b6d
e100c479d9fc22f8
```

## Interpretation

Alive:

- The wrapper-level runtime accept/fallback machinery is useful for future
  gates because it preserves raw and gated artifacts separately.
- The raw seed-0 dynalign mechanism remains a mechanism probe.

Killed:

- `selector_gap_min_ge_q0p7_numeric_changed` as a publishable method. The score
  is not source-specific enough; fresh zero/shuffled controls pass the same
  threshold and retain most of the apparent wins.

Weakened:

- The prior offline replay. It was directionally useful for designing the
  runtime gate, but it overestimated control safety because the control
  artifacts were not fresh raw-control generations under the frozen policy.

Blocked:

- Seed repeats and cross-family widening for this policy. Running seed 3 would
  not rescue a policy that already fails seed-0 controls.

## Next Exact Gate

Do not widen this branch. The next highest-value method branch is a
control-contrastive learned innovation connector:

- freeze source and target,
- train a small Q-former/Perceiver-style bottleneck or additive innovation
  sidecar,
- include matched source, zero source, and shuffled source in the objective,
- explicitly penalize zero/shuffled deltas away from target-alone behavior,
- rerun the same GSM70 seed-0 source-control gate before any seed or benchmark
  widening.

If that branch also retains wins under zero/shuffled controls, demote learned
same-family connectors on this task and move to either a stronger-source pair
or a cross-family falsification-first benchmark.

## Verification

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_gsm8k_contract_residual_sweep.py \
  tests/test_analyze_gsm8k_accept_fallback.py \
  tests/test_analyze_gsm8k_source_controls.py
```

Result: `44 passed in 0.12s`.
