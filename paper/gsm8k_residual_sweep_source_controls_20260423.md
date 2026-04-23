# GSM8K Residual Sweep Source-Control Wrapper

Date: 2026-04-23

## Status

This is a reproducibility and falsification-gate update, not a new positive
method result. The latest query-innovation resampler row remains demoted because
its only live win survived zero-source and shuffled-source controls.

## Change

`scripts/run_gsm8k_contract_residual_sweep.py` now has an opt-in
`--run-source-controls` mode. For each live row that passes the normal sweep
contract, the runner:

- evaluates a matched zero-source control with `--source-kv-control zero`
- evaluates a matched shuffled-source control with
  `--source-prompt-control shuffle_examples`
- runs `scripts/analyze_gsm8k_source_controls.py` on the live and control
  predictions
- stores row-local source-control artifacts under
  `{results_dir}/{label}/source_controls/`
- blocks markdown promotion unless the source-control analyzer returns
  `source_controls_support_matched_source_signal`

Rows that fail the normal live contract record
`not_run_live_gate_failed` instead of spending control compute.

## Verification

Focused test command:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_gsm8k_contract_residual_sweep.py \
  tests/test_analyze_gsm8k_source_controls.py
```

Result: `39 passed in 0.13s`.

The tests cover:

- parsing the new source-control flags
- exact zero-source and shuffled-source command construction
- analyzer invocation with labeled controls
- target-fallback analyzer flag propagation
- markdown promotion blocking when source controls fail

## Subagent Notes

The method-side subagent recommended using this wrapper before further method
branching, then testing a source-control-aware accept/fallback gate on the
GSM70 `dynalign + resid16` lane. The repo-audit subagent flagged
`scripts/build_gsm8k_contract_manifest.py` as a follow-up cleanup because it
still resolves live/control rows positionally instead of by explicit label.

## Decision

The wrapper makes source-control validation first-class in the residual sweep
path. This does not revive the demoted query-innovation row; it prevents the
same false-positive pattern from being promoted again.

The next exact gate is to run the integrated wrapper on the strongest remaining
real lane, starting with GSM70 `dynalign_module_replace_residrank16` seed 0 and
one finite repeat, then decide whether a source-control-aware accept/fallback
gate has enough real signal to justify implementation.
