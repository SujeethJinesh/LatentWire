# GSM8K32 Anchor-Tail Seed-1 Falsification

Date: `2026-04-22`

## Why This Note Exists

The four-seed GSM70 stability read promoted one narrow codec-side follow-up:
make `bridge_ridge_qk_dynalign_anchor_tail_module_replace` truly `V`-only, keep
`K` on the live dynalign module path, and preserve only a small exact value-side
anchor while quantizing the residual `V` tail.

Before spending a full GSM70 rerun on that branch, this note records the
cheapest bad-seed falsification check on the frozen GSM8K32 contract.

## Code Change Under Test

The branch was tightened in-repo so that:

1. `quant_preserve_proj_K` stays zero for
   `bridge_ridge_qk_dynalign_anchor_tail_module_replace`
2. `K` returns the live module prediction unchanged at runtime
3. `V` applies anchor-tail quantization only to the residual against the base
   bridge:

   `base_v + exact((module_pred_v - base_v) @ P) + Q((module_pred_v - base_v) @ (I - P))`

This is the smallest faithful implementation of the intended selective-precision
story from `references/446_anchor_selective_precision_followup_refs.md`.

## Run

Command:

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_anchor_tail_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_v_anchor_tail_seed1_20260422 \
  --seed 1
```

Artifacts:

- `.debug/gsm8k32_v_anchor_tail_seed1_20260422/gsm8k_contract_residual_sweep_20260421.md`
- `.debug/gsm8k32_v_anchor_tail_seed1_20260422/gsm8k_contract_residual_sweep_20260421.json`
- `checkpoints/gsm8k_contract_residual_sweep_20260421/dynalign_anchor_tail_module_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_anchor_tail_module_replace_cal64_chat_seed1.pt.health.json`

## Result

On the frozen `32`-example same-pair slice, seed `1` is still a clean
checkpoint-health failure:

| Seed | Status | Accuracy | Correct | Numeric coverage | Empty preds | Non-finite | First bad key |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `checkpoint_nonfinite` | `0.0000` | `0` | `0/32` | `32` | `2381056` | `W_V.8` |

The quarantined checkpoint repeats the same failure family as the live
`dynalign_module_replace_residrank16` bad seeds:

- `W_V.8`
- `quant_proj_V.8`
- `quant_aux_proj_V.8`
- `quant_query_resid_V_left.8`
- `quant_query_resid_V_right.8`
- `quant_query_aux_resid_V_left.8`
- `quant_query_aux_resid_V_right.8`
- `quant_query_module_slots.8`

## Interpretation

This runtime-only `V`-side anchor-tail repair does **not** move the failure
surface upstream.

- The checkpoint still becomes non-finite before evaluation.
- The collapse remains localized to the same layer-8 `V` tensor family.
- So this branch is not yet evidence for robust selective precision; it is
  evidence that a wrapper around the current `V` prediction is too late if the
  calibration fit itself is already unstable.

## Updated Branch Read

Treat the current `V`-only anchor-tail result as:

1. a useful implementation cleanup
2. a clean seed-1 falsification of the wrapper-only hypothesis
3. not worth widening to more seeds until the layer-8 `V` fit itself is
   stabilized

## Next Exact Gate

The next bounded robustness branch should move one level earlier than the
current wrapper:

1. patch the layer-8 `V` calibration fit itself
   - stronger regularization / clipping / whitening / per-layer fallback, or
   - a smaller layer-local value-only repair path
2. rerun the same bad-seed check first
3. only if that stays finite, reopen the full GSM70 seed audit
