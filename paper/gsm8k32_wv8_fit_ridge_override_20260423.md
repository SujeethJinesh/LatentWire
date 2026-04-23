# GSM8K32 W_V.8 Fit Ridge Override

- date: `2026-04-23`
- live branch: `dynalign_module_replace_residrank16`
- gate: stabilize bad-seed layer-8 value fit without erasing the live row
- source -> target: `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B`
- frozen slice: GSM8K32, exact target ID parity

## Motivation

The previous bad-seed failures localized to `W_V.8`. Simple source/target
whitening made the checkpoints finite, but every simple whitening scope either
tied target or damaged seed `0`. The smallest remaining diagnostic was a direct
fit-time ridge override on only the target layer-8 value stream.

## Implementation

Commit `fba00f40` adds a generic `TranslatorConfig` override for the
closed-form source-to-target alignment fit:

- `fit_ridge_override_lambda`
- `fit_ridge_override_streams`
- `fit_ridge_override_layers`

The override is wired through `latent_bridge/calibrate.py` and
`scripts/run_gsm8k_contract_residual_sweep.py`, with checkpoint suffixes and
row provenance so override artifacts cannot silently reuse baseline
checkpoints.

Focused tests passed:

- `tests/test_translator_core.py -k 'fit_ridge_override or selective_conditioning or selective_source_whitening or selective_target_whitening'`
- `tests/test_calibrate_and_ablation.py -k 'unit_tested_ablation_flags'`
- `tests/test_gsm8k_contract_residual_sweep.py -k 'fit_ridge_override or selective_conditioning or whitening_flags or checkpoint_path_fit_ridge'`
- `tests/test_gsm8k_contract_residual_sweep_runtime_helpers.py`

## Run

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_dynalign_resid16_seed1_wv8ridge_lam0p01_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_dynalign_resid16_seed1_wv8ridge_lam0p01_20260423 \
  --seed 1 \
  --fit-ridge-override-lambda 1e-2 \
  --fit-ridge-override-streams v \
  --fit-ridge-override-layer 8
```

Artifacts:

- results JSON: `.debug/gsm8k32_dynalign_resid16_seed1_wv8ridge_lam0p01_20260423/gsm8k_contract_residual_sweep_20260421.json`
- markdown readout: `.debug/gsm8k32_dynalign_resid16_seed1_wv8ridge_lam0p01_20260423/gsm8k_contract_residual_sweep_20260421.md`
- checkpoint: `.debug/checkpoints_gsm8k32_dynalign_resid16_seed1_wv8ridge_lam0p01_20260423/dynalign_module_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_module_replace_cal64_chat_fitridgev_layers8_lam0p01_seed1.pt`

## Result

Seed `1`, targeted `W_V.8` ridge lambda `1e-2`:

- status: `ok`
- accuracy: `0.0625`
- numeric coverage: `32 / 32`
- empty predictions: `0`
- paired vs target: `0` win, `0` loss, `32` tie
- checkpoint nonfinite values: `0`
- first bad checkpoint key: none
- checkpoint max abs: `6416.1553`
- top abs tensor: `quant_aux_proj_V.15`, not `W_V.8`

## Interpretation

This weakens the direct ridge-regularization hypothesis. The branch proves that
a localized `W_V.8` ridge increase can remove the catastrophic nonfinite
checkpoint failure, but it does not recover a positive communication signal on
the frozen bad seed. Because the row exactly ties target with full coverage,
running seed `0` or GSM70 would not clear the live stability gate.

Current status:

- alive: the broader layer-8 value-fit localization and the seed-0 live clue
- weakened: direct `W_V.8` ridge damping as the stabilizing method
- saturated: simple whitening and direct ridge-only stabilization
- blocked: stable positive same-pair evidence beyond the fragile seed-0 row

## Next Gate

Do not continue scalar ridge sweeps unless there is a new diagnostic showing
where the signal survives. The next highest-value branch is a value-innovation
codec for layer `8`: protect a small set of high-sensitivity `V` channels or
directions, regularize only the residual tail, and report byte accounting. The
minimal decisive test is still GSM8K32 seed `1` first, then seed `0` only if
seed `1` is finite and positive.
