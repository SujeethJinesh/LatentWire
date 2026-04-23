# GSM8K32 V8 Outlier Escrow And Source-Only Layer-8 K/V Conditioning

Date: 2026-04-23
Commit: `98fb3d4ac58353dec6801c03c4215f1fe2e090ff`

## Gate

The live `dynalign_module_replace_residrank16` lane remains blocked by
bad-seed layer-8 value-side non-finites. The gate for this screen was:

1. implement a protected layer-8 value-channel outlier-escrow branch;
2. run GSM8K32 seed `1`;
3. only run seed `0` if seed `1` is finite and positive;
4. if escrow fails, try the cheapest no-code fallback: source-only layer-8
   `K/V` conditioning.

## V8 Outlier-Escrow Branch

Branch:
`bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace`

Harness base:
`dynalign_v8_outlier_escrow_module_replace`

Command:

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_v8_outlier_escrow_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_dynalign_v8_outlier_escrow_resid16_seed1_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_dynalign_v8_outlier_escrow_resid16_seed1_20260423 \
  --seed 1
```

Artifacts:

- Summary JSON:
  `.debug/gsm8k32_dynalign_v8_outlier_escrow_resid16_seed1_20260423/gsm8k_contract_residual_sweep_20260421.json`
- Summary JSON SHA256:
  `bf4d5a900f71132ff24ac8fc5d741c9e71bc5f13b9cf68e19a66bc490241b47b`
- Summary MD:
  `.debug/gsm8k32_dynalign_v8_outlier_escrow_resid16_seed1_20260423/gsm8k_contract_residual_sweep_20260421.md`
- Summary MD SHA256:
  `cf1c546c199be39f8a03be488d7d444a1f855b2f2ffb76e414f464a00c2dd090`
- Health JSON:
  `.debug/checkpoints_gsm8k32_dynalign_v8_outlier_escrow_resid16_seed1_20260423/dynalign_v8_outlier_escrow_module_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_v8_outlier_escrow_module_replace_cal64_chat_seed1.pt.health.json`
- Health JSON SHA256:
  `5ad35fa53ccb3f98297531e901613a895a170619081492787071175e73caba01`
- Calibration SHA256:
  `462d9313b8f0a8bd310860484ddc3d1fa128e5a3af8256ed409add554c33984a`

Result:

- status: `checkpoint_nonfinite`
- checkpoint first bad key: `W_V.8`
- checkpoint nonfinite numel: `2381056`
- checkpoint max abs: `6416.1553`
- nonfinite family: `W_V.8`, `quant_proj_V.8`, `quant_aux_proj_V.8`,
  `quant_query_resid_V_left.8`, `quant_query_resid_V_right.8`,
  `quant_query_aux_resid_V_left.8`, `quant_query_aux_resid_V_right.8`,
  `quant_query_module_slots.8`

Read:

The protected layer-8 value-channel branch does not move the numerical failure
upstream. It fails at the same `W_V.8` surface before evaluation, so seed `0`
was not run. Treat this branch as killed unless we later add a direct
layer/stream-specific fit regularizer.

## Source-Only Layer-8 K/V Conditioning

Branch:
`dynalign_module_replace_residrank16` with source whitening only on target
layer `8`, streams `K/V`.

Command:

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_dynalign_resid16_seed1_srcwhitekv_l8_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_dynalign_resid16_seed1_srcwhitekv_l8_20260423 \
  --seed 1 \
  --whitening \
  --whitening-streams kv \
  --conditioning-target-layer 8
```

Artifacts:

- Summary JSON:
  `.debug/gsm8k32_dynalign_resid16_seed1_srcwhitekv_l8_20260423/gsm8k_contract_residual_sweep_20260421.json`
- Summary JSON SHA256:
  `385ec585620774f6c09a65a2824b743fa36cb381aade9aef7c9a28c435dcb2cb`
- Summary MD:
  `.debug/gsm8k32_dynalign_resid16_seed1_srcwhitekv_l8_20260423/gsm8k_contract_residual_sweep_20260421.md`
- Summary MD SHA256:
  `b4e128766096382d7cd2e9a6e236158f3c0103d8ce2fb81ade3d6d0cfcdfe0ae`
- Prediction JSONL:
  `.debug/gsm8k32_dynalign_resid16_seed1_srcwhitekv_l8_20260423/dynalign_module_replace_residrank16.jsonl`
- Prediction JSONL SHA256:
  `418b2a604fd4937b511cbbbb907a9515cf04930aa2cc32e21c03a94daa667e4e`
- Prediction meta SHA256:
  `2fe7372ea4726809c586d4fa9d0a9b605b272f8c4e3423cbfb846a87548d25e9`
- Calibration SHA256:
  `462d9313b8f0a8bd310860484ddc3d1fa128e5a3af8256ed409add554c33984a`

Result:

- status: `ok`
- accuracy: `0.0625` (`2/32`)
- numeric extraction coverage: `32/32`
- empty predictions: `0`
- checkpoint nonfinite numel: `0`
- paired vs target: `0` win, `0` loss, `32` tie
- beats target: `false`

Read:

Source-only layer-8 `K/V` conditioning fixes the seed-1 checkpoint failure but
does not produce positive communication on the frozen 32-example surface. Since
it only ties target on seed `1`, seed `0` was not run. This weakens the
hypothesis that simple conditioning scope is the stabilizing method; target
whitening was not the only reason the prior conditioned rows failed.

## Decision

- Killed: `bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace` as
  currently implemented.
- Weakened: source-only layer-8 `K/V` conditioning as a promotable robustness
  method.
- Still alive: the localization claim that the blocker is the layer-8
  value-side fit surface.
- Next exact gate: implement a targeted layer/stream-specific `W_V.8`
  regularization or diagnostics branch, then run GSM8K32 seed `1` before any
  seed `0`, GSM70, or cross-family widening.
