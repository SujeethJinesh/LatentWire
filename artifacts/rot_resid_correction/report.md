# C5 Rotated Residual Correction Gate

## Decision

Status: `KILL_TOP8X32_MOE_RESIDUAL_DIAGNOSTIC`.

The missing residual artifacts were added and a one-trace Granite diagnostic was
run. The top-8-module, 32-column-per-module MoE residual correction worsened the
known Granite tail trace relative to tight ParoQuant, so it is not promoted to
3-trace smoke under the current design. See
`artifacts/rot_resid_correction/residual_smoke_report.md`.

## Method

After ParoQuant quantization, define:

`DeltaW = W_fp - W_pq`

For an activation row `x`, use a protected column subset `P` and compute:

`y = x W_pq + x_P DeltaW_P`

This is a correction on top of the ParoQuant baseline, not a replacement for
ParoQuant. Candidate columns are selected by:

`score_i = EMA(x_i^2) * ||DeltaW_i||_2^2`

where `i` indexes input columns of an eligible linear/expert tensor. The
optional KLLOOK pool may add columns whose one-at-a-time correction lowers
`KL(p_BF16 || p_Q)` on calibration traces.

## Why This Branch Exists

V1 showed ParoQuant dominates Nemotron M11b and Granite already had a strong
ParoQuant result. The remaining positive-method question is whether a cheap
residual correction can improve ParoQuant tails or rescue models where rotation
alone is weak. Granite has an obvious tail trace:

- prompt index 4: ParoQuant recovery `-26.37` with a small positive static gap.

That trace is the right first smoke target if residual tensors are available.

## Cache Produced

The branch now has:

- residual column norms:
  `artifacts/rot_resid_correction/residual_cache_granite_tight_20260528T2025Z/`;
- activation EMA on the tail trace:
  `artifacts/rot_resid_correction/activation_ema_granite_tail4_top8_20260528T2030Z/`;
- candidate pool:
  `artifacts/rot_resid_correction/residual_candidate_pool_tail4_top8.json`;
- selected `DeltaW[:, P]` tensors:
  `artifacts/rot_resid_correction/delta_columns_granite_tail4_top8x32_20260528T2048Z/`.

The selected delta-column cache is about 54 MiB working set across eight Granite
MoE input projections.

## Gate

Do not launch the 3-trace residual-correction smoke for this top-8x32 MoE
candidate. A future residual-correction branch needs a new design gate, such as
coefficient-shrunk correction or KLLOOK-gated columns, before using GPU time.
