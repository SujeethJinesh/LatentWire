# GSM8K70 Seed Stability Partial Read

Date: `2026-04-22`

## Why This Note Exists

The highest-priority gate after the first larger frozen same-pair success was
seed stability. The first repeat seed is now complete, and the result is
strong enough to change the paper story even before all seeds finish.

## Current Seeds

On the same frozen `70`-example same-pair slice:

| Seed | Row | Accuracy | Correct | Numeric coverage | Read |
|---:|---|---:|---:|---:|---|
| 0 | `dynalign_module_replace_residrank16` | 0.1143 | 8 | 70 | live positive seed |
| 1 | `dynalign_module_replace_residrank16` | 0.0000 | 0 | 0 | degenerate collapse |

Seed `1` is not an ordinary regression. The generated outputs are repeated
`!` tokens, and the row fails numeric extraction coverage completely.

## Interpretation

1. The live same-pair lane is currently **not** seed-stable.
2. This is more severe than a small performance drop.
   - The repeat seed produced a degenerate checkpoint or degenerate translated
     outputs.
3. The current main gap is therefore no longer only “replicate the gain.”
   - It is now “understand and eliminate calibration / seed fragility.”

## Current Diagnosis

The seed `1` checkpoint is numerically corrupted rather than merely weak:

- seed `0` checkpoint: `0` non-finite values
- seed `1` checkpoint: `2,381,056` non-finite values
- the largest observed divergence is in `quant_proj_K.1`

So the current failure mode should be treated as a calibration robustness bug
or instability, not just benchmark variance.

## What This Does To The Story

Before this partial read, the larger-slice story was:

- the live row survives beyond GSM8K32

After this partial read, the story becomes:

- the live row can survive beyond GSM8K32
- but the method is currently fragile enough across seeds that it is nowhere
  near publication-ready

## Immediate Next Gate

1. finish the remaining repeat seed(s)
2. determine whether the failure is:
   - true method instability
   - calibration instability
   - or a checkpoint-generation bug
3. only after that, decide whether to invest in cross-family testing or move
   straight to robustness fixes
