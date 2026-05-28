# C5 Rotated Residual Correction Gate

## Decision

Status: `NEEDS_WEIGHT_RESIDUAL_CACHE`.

Rotated residual correction is well-defined, but the current committed artifacts
do not include full-precision weights, ParoQuant-dequantized weights, or
per-layer residual norms. Without those tensors we cannot produce a real
candidate column pool. This artifact therefore provides the exact method,
candidate-pool template, and PyTorch reference only.

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

## Required Cache

To run this branch, the next ParoQuant smoke must optionally write:

- eligible tensor names and shapes;
- `W_fp` or column norms of `W_fp`;
- `W_pq` or column norms of `DeltaW = W_fp - W_pq`;
- post-rotation/dequant tensor identity matching the forward pass;
- calibration activation EMA per tensor input column;
- selected candidate top-k columns per tensor.

The full tensors do not need to be preserved if the candidate pool stores
`||DeltaW_i||_2^2`, activation scores, and top-k indices.

## Gate

Do not launch a residual-correction GPU run until candidate pools exist. Once
they exist, run Granite only on:

1. tail trace `opencompass_AIME2025_I_4`,
2. weak positive trace `opencompass_AIME2025_I_2`,
3. representative positive trace `opencompass_AIME2025_I_8`.

Promote if it improves the tail without losing more than 0.05 recovery on the
representative trace, or wins at least two of the three traces against
ParoQuant.
