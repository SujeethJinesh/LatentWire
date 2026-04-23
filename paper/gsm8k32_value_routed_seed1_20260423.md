# GSM8K32 Value-Routed Seed-1 Falsification

- date: `2026-04-23`
- live branch family: learned slotted/query module replacement
- branch tested: `dynalign_value_routed_module_replace_residrank16`
- gate: determine whether the existing learned value-gated connector surface
  can make a second seed finite before implementing a new query-resampler
  variant
- source -> target: `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B`
- frozen slice: GSM8K32, exact target ID parity expected

## Why This Test

The closed-form value-side repair lane is now mostly exhausted:

- wrapper-only V anchor/tail repeats the `W_V.8` nonfinite failure
- scalar `W_V.8` ridge and protected-ridge variants become finite but collapse
  to target parity
- source-only layer-8 `K/V` conditioning is finite but only ties target

Before adding a new named learned query/resampler connector, this test checks
the nearest existing learned surface. The value-routed branch already preserved
the GSM8K32 seed-0 live row:

- `dynalign_value_routed_module_replace_residrank16 = 4/32`
- full numeric coverage
- paired vs target: `2` wins / `0` losses / `30` ties

So the cheapest decisive question is whether the same learned V-gated connector
also fixes seed `1`.

## Run

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_value_routed_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_dynalign_value_routed_resid16_seed1_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_dynalign_value_routed_resid16_seed1_20260423 \
  --seed 1
```

Artifacts:

- summary JSON:
  `.debug/gsm8k32_dynalign_value_routed_resid16_seed1_20260423/gsm8k_contract_residual_sweep_20260421.json`
- summary JSON SHA256:
  `9a29f7357a230ccaaf3ae9992d090c4207c3ce917fb893b27f36a68bb4f12fa7`
- summary MD:
  `.debug/gsm8k32_dynalign_value_routed_resid16_seed1_20260423/gsm8k_contract_residual_sweep_20260421.md`
- summary MD SHA256:
  `f3d30b489037c0553e4ab3c44a5accfcace0da23a75168281000b1d11e2cb761`
- checkpoint health JSON:
  `.debug/checkpoints_gsm8k32_dynalign_value_routed_resid16_seed1_20260423/dynalign_value_routed_module_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_value_routed_module_replace_cal64_chat_seed1.pt.health.json`
- checkpoint health JSON SHA256:
  `010214b3447ba2b31d7fb17c5001f2d5b57ed3879d4cea0fb3d89c85bb0ae575`

## Result

| Seed | Status | Accuracy | Numeric coverage | Empty preds | Non-finite | First bad key |
|---:|---|---:|---:|---:|---:|---|
| 1 | `checkpoint_nonfinite` | `0.0000` | `0/32` | `32` | `2,382,081` | `W_V.8` |

The nonfinite family repeats the same layer-8 value-side failure:

- `W_V.8`
- `quant_proj_V.8`
- `quant_aux_proj_V.8`
- `quant_query_resid_V_left.8`
- `quant_query_resid_V_right.8`
- `quant_query_aux_resid_V_left.8`
- `quant_query_aux_resid_V_right.8`
- `quant_query_module_slots.8`

## Interpretation

This is a clean falsification of the current value-routed branch as a
seed-stability rescue. The branch remains a useful seed-0 control because it
preserves the live GSM8K32 row, but it does not solve the `W_V.8` checkpoint
health problem on seed `1`.

Current branch statuses:

- alive: seed-0 matched-source non-copy signal; learned connector family as the
  right architectural class
- killed/saturated: rerunning existing value-routed variants as the next
  stability fix
- blocked: seed-stable positive same-pair evidence
- promoted: an explicit guarded query/resampler connector with fit-time
  finite/norm checks and fallback-to-base for bad layer outputs

## Next Exact Gate

Do not run `dynalign_value_routed_module_replace` on GSM70 or cross-family
pairs. The branch fails the cheap seed-1 checkpoint-health screen.

The next implementation branch should be a new named connector, e.g.
`bridge_ridge_qk_dynalign_query_resampler_replace`, that reuses the existing
slotted query-module plumbing but adds the evaluation contract from the start:

1. explicit bottleneck knobs: query slots via `bridge_bank_size`, rank via
   `quantization_correction_rank`
2. fit-time finite/norm checks before checkpoint materialization
3. fallback-to-base or zero-sidecar behavior for layers whose learned value
   module becomes nonfinite
4. matched, zero-source, and shuffled-source target-safe controls
5. capacity/null sweep across `0/4/16/32` slots before any cross-family claim

This maps directly onto the learned connector prior from BLIP-2/Q-Former
(https://arxiv.org/abs/2301.12597), Flamingo/Perceiver Resampler
(https://arxiv.org/abs/2204.14198), Perceiver IO
(https://arxiv.org/abs/2107.14795), and the direct cache-fusion competitor C2C
(https://arxiv.org/abs/2510.03215). The conditional side-information framing
should use the Wyner-Ziv intuition: transmit the source innovation relative to
the target's own cache rather than the full source state.
