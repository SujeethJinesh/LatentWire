# GSM8K32 W_V.8 Protected Fit Ridge

- date: `2026-04-23`
- live branch: `dynalign_module_replace_residrank16`
- gate: preserve value-side information while stabilizing bad-seed `W_V.8`
- source -> target: `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B`
- frozen slice: GSM8K32, exact target ID parity

## Motivation

The scalar `W_V.8` fit ridge override made seed `1` finite but collapsed to
target parity. This follow-up tested whether a tiny protected innovation set
could keep the stabilizing tail ridge while preserving high-signal value
channels at the base ridge.

## Implementation

Commits:

- `4d71704f`: added `fit_ridge_protected_rank`, calibrate CLI, residual-sweep
  provenance, and a protected residual split.
- `a3bc0509`: replaced the unsafe split-tail solve with a safer overwrite:
  fit the full residual at the stabilizing lambda, then overwrite only the
  protected innovation columns with the base-lambda solve.

The protected mask is selected from the grouped-transport residual target
`Y - base_pred`, not raw `Y`, so the branch targets innovation/outlier columns
rather than common high-magnitude target directions.

Focused tests passed:

- `tests/test_translator_core.py -k 'protected_outputs or top_output_mask or fit_ridge_override'`
- `tests/test_translator_core.py -k 'fit_ridge_override or protected_outputs or top_output_mask or selective_conditioning'`
- `tests/test_calibrate_and_ablation.py -k 'unit_tested_ablation_flags'`
- `tests/test_gsm8k_contract_residual_sweep.py -k 'fit_ridge_override or whitening_flags'`
- `tests/test_gsm8k_contract_residual_sweep_runtime_helpers.py`

## Runs

### Unsafe Split, Protected Rank 2

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_dynalign_resid16_seed1_wv8_protect2_lam0p01_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_dynalign_resid16_seed1_wv8_protect2_lam0p01_20260423 \
  --seed 1 \
  --fit-ridge-override-lambda 1e-2 \
  --fit-ridge-override-streams v \
  --fit-ridge-override-layer 8 \
  --fit-ridge-protected-rank 2
```

Result:

- status: `checkpoint_nonfinite`
- first bad key: `W_V.8`
- checkpoint nonfinite values: `2,380,800`
- interpretation: splitting the tail residual solve is numerically unsafe.
  The nonfinites concentrated in the unprotected tail columns, so this variant
  was replaced by the overwrite version.

### Overwrite, Protected Rank 2

Artifacts:

- results JSON: `.debug/gsm8k32_dynalign_resid16_seed1_wv8_protect2_overwrite_lam0p01_20260423/gsm8k_contract_residual_sweep_20260421.json`
- checkpoint: `.debug/checkpoints_gsm8k32_dynalign_resid16_seed1_wv8_protect2_overwrite_lam0p01_20260423/dynalign_module_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_module_replace_cal64_chat_fitridgev_layers8_lam0p01_protect2_seed1.pt`

Result:

- status: `ok`
- accuracy: `0.0625`
- numeric coverage: `32 / 32`
- empty predictions: `0`
- paired vs target: `0` win, `0` loss, `32` tie
- checkpoint nonfinite values: `0`
- first bad key: none

### Overwrite, Protected Rank 4

Artifacts:

- results JSON: `.debug/gsm8k32_dynalign_resid16_seed1_wv8_protect4_overwrite_lam0p01_20260423/gsm8k_contract_residual_sweep_20260421.json`
- checkpoint: `.debug/checkpoints_gsm8k32_dynalign_resid16_seed1_wv8_protect4_overwrite_lam0p01_20260423/dynalign_module_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_module_replace_cal64_chat_fitridgev_layers8_lam0p01_protect4_seed1.pt`

Result:

- status: `ok`
- accuracy: `0.0625`
- numeric coverage: `32 / 32`
- empty predictions: `0`
- paired vs target: `0` win, `0` loss, `32` tie
- checkpoint nonfinite values: `0`
- first bad key: none

## Interpretation

The protected-overwrite implementation is a better numerical primitive than
the unsafe split-tail solve: it preserves finite checkpoints for protected
ranks `2` and `4`. However, both protected ranks exactly reproduce scalar
ridge parity on seed `1`.

This weakens the hypothesis that a tiny scalar protected-ridge channel can
recover the lost bad-seed communication signal. It is still useful
infrastructure for future byte-aware value codecs, but it is not a promotable
method branch by itself.

Current status:

- alive: layer-8 value localization and the need for value-side innovation
  coding
- weakened: scalar protected ridge as the stabilizing method
- killed: split-tail protected residual solve
- saturated: scalar ridge-only, simple whitening, wrapper-only escrow
- blocked: stable positive same-pair evidence on bad seeds

## Next Gate

Do not run seed `0`, GSM70, or cross-family widening for protected scalar ridge:
seed `1` is finite but not positive. The next highest-value gate is either:

1. a diagnostic/source-correctness and flip-table audit on the live seed-0 wins
   to determine whether the positive clue is source-answer copying, target-cache
   regularization, or actual communication; or
2. a learned query/resampler connector with an explicit bottleneck, treating
   the current fit-ridge work as evidence that closed-form scalar fixes are
   exhausted.
