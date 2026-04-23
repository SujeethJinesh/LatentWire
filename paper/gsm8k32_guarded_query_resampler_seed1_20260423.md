# GSM8K32 Guarded Query-Resampler Seed-1 Gate

- date: `2026-04-23`
- branch family: learned slotted/query module replacement with fit-time guards
- branch tested: `dynalign_query_resampler_replace_residrank16`
- correction: `bridge_ridge_qk_dynalign_query_resampler_replace`
- gate: determine whether a guarded query-resampler can eliminate seed-1
  nonfinite checkpoint failures without breaking GSM8K32 validity
- source -> target: `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B`
- frozen slice: GSM8K32, exact target ID parity required

## Why This Test

The live same-family clue remains seed-fragile. The prior seed-1 screens for
anchor-tail and value-routed learned variants failed before evaluation with
nonfinite layer-8 value tensors, especially `W_V.8`. This test implements the
next branch named in the ledger: reuse the existing slotted query-module path,
but guard alignment/module tensors before checkpoint materialization so bad
seeds degrade to finite zero-sidecar behavior instead of poisoning the
checkpoint.

## Implementation

Code changes:

- Added `bridge_ridge_qk_dynalign_query_resampler_replace` to calibration,
  translator fit/runtime, evaluation live-query-head dispatch, and the GSM8K
  residual sweep registry.
- Added branch-specific fit guards for `W_K`, `W_V`, module slots, module
  projections, hidden, and output tensors. Any nonfinite or extremely high-norm
  tensor is replaced with a zero fallback before copying into checkpoint state.
- Added prediction sidecar provenance for translator correction, rank, and
  bridge bank size.
- Added regression tests for parser registration, checkpoint path construction,
  nonfinite tensor fallback, and evaluation live-query-head collection.

## Run

Initial run:

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_query_resampler_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_guarded_query_resampler_seed1_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_guarded_query_resampler_seed1_20260423 \
  --seed 1
```

This produced a finite checkpoint, then exposed a missing evaluation registration:
`bridge_ridge_qk_dynalign_query_resampler_replace requires live query heads`.
The evaluation dispatch was fixed and rerun against the same checkpoint:

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_query_resampler_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_guarded_query_resampler_seed1_evalfix_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_guarded_query_resampler_seed1_20260423 \
  --seed 1
```

## Artifacts

- checkpoint:
  `.debug/checkpoints_gsm8k32_guarded_query_resampler_seed1_20260423/dynalign_query_resampler_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_query_resampler_replace_cal64_chat_seed1.pt`
- checkpoint SHA256:
  `b4accaec6a9b6414015c58017a0b77894be88462898654b0261b690a9d7653c7`
- summary JSON:
  `.debug/gsm8k32_guarded_query_resampler_seed1_evalfix_20260423/gsm8k_contract_residual_sweep_20260421.json`
- summary JSON SHA256:
  `a61ff8f9ec5b3deecb08ba7ca22671164438aa6a24a37bc276a3fec19a9ad290`
- summary MD:
  `.debug/gsm8k32_guarded_query_resampler_seed1_evalfix_20260423/gsm8k_contract_residual_sweep_20260421.md`
- summary MD SHA256:
  `27dd9eaa100825d90cf4add1cc10a4d8b5f7553cf31b6151768dd0e675dc5b34`
- prediction sidecar:
  `.debug/gsm8k32_guarded_query_resampler_seed1_evalfix_20260423/dynalign_query_resampler_replace_residrank16.jsonl.meta.json`
- prediction sidecar SHA256:
  `e23fcdcef3eb863ae7e6f7accd69fbe0953fca095a0a325f66d4e12b16c9a380`

## Result

| Seed | Status | Accuracy | Paired vs target | ID parity | Numeric coverage | Empty preds | Nonfinite | First bad key |
|---:|---|---:|---|---|---:|---:|---:|---|
| 1 | `ok` | `2/32` | `0` wins / `0` losses / `32` ties | pass | `32/32` | `0` | `0` | `-` |

Checkpoint health:

- `checkpoint_nonfinite_numel = 0`
- `checkpoint_first_bad_key = null`
- `checkpoint_max_abs = 6416.1553`
- largest tensor: `quant_aux_proj_V.15`
- translator provenance in prediction sidecar:
  - `translator_quantization_correction = bridge_ridge_qk_dynalign_query_resampler_replace`
  - `translator_quantization_correction_rank = 16`
  - `translator_bridge_bank_size = 4`

Validation status:

- row count: pass
- exact example-id parity vs target: pass
- numeric extraction coverage: pass
- empty predictions: pass
- beats target: fail

## Interpretation

The guarded query-resampler clears the immediate checkpoint-health and validity
gate that killed the prior seed-1 learned variants. It is therefore useful as a
safety wrapper and as the correct surface for future learned connector work.

It does not yet clear the positive-method gate. On the decisive seed-1 GSM8K32
slice it ties target exactly: no candidate-only wins and no target losses. This
means the guard likely suppresses the unstable value-side signal into a
target-parity/fallback regime rather than recovering source communication.

Current branch status:

- alive: guarded slotted/query-resampler as the finite learned connector surface
- weakened: query-resampler-as-implemented as a positive method, because seed 1
  is target parity
- killed/saturated: rerunning existing value-routed/anchor-tail variants as
  seed-stability fixes
- blocked: seed-stable positive matched-source evidence

## Next Exact Gate

Do not widen this branch to GSM70 or cross-family yet. The next decisive move is
one of:

1. Add an innovation/residual-only target-safe query-resampler path that predicts
   bounded source innovation relative to the target cache rather than full
   replacement. Minimal gate: GSM8K32 seed1 must remain finite and produce at
   least one candidate-only win without target losses.
2. Run a cheap capacity/null sweep on the current guarded surface
   (`bridge_bank_size = 0/4/16/32`, rank 16) to determine whether target parity
   is caused by insufficient slot capacity or by the guard suppressing all
   source signal. Minimal gate: any capacity beats target on seed1 while keeping
   checkpoint health and full coverage.
3. If both fail, promote output-aware training for the learned connector rather
   than further closed-form value-side repair.

The current method is not paper-ready, but the failure is now clean: finite,
valid, seed-1 target parity rather than nonfinite checkpoint corruption.
