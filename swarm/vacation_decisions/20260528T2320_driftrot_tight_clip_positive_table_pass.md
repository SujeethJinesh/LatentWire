# DriftRot Tight-Clip Granite Positive-Trace Table

Date: 2026-05-28T23:20Z

## Decision

`PASS_TIGHT_CLIP_GRANITE_POSITIVE_TRACES`

Tight clip `[0.5, 2.0]` is promoted as Tier-2 DriftRot tail-control evidence
for Granite. It is not the headline method by itself, but it is now stronger
than an inspected tail anecdote.

## Evidence

Source table:
`artifacts/scale_cvar_clip/granite_tight_positive_trace_table.csv`

Runs:

- inspected screen:
  `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z`
- held-out split:
  `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_holdout_20260528T2144Z`
- final positive trace:
  `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_finalpos8_20260528T2251Z`

All eight recoverable Granite traces:

| Metric | Original ParoQuant | Tight clip |
|---|---:|---:|
| Median recovery | 0.754 | 0.922 |
| CI95 | [0.477, 1.004] | [0.781, 1.647] |
| Worst trace | -26.37 | 0.601 |
| Mean recovery | -2.532 | 1.581 |

Median per-trace margin: `+0.0509`, CI95 `[0.0166, 0.370]`.

## Interpretation

The result supports a narrow DriftRot claim: a drift/tail-aware clip objective
can robustify a static rotation baseline in the Granite dense-hybrid regime.
This should not be framed as "ParoQuant with tuned hyperparameters" or as the
headline method. It belongs as tail-control evidence inside the regime-aware
rotation story.

## Next Gate

Either integrate this into the paper as a Tier-2 positive result, or test
another model/tail surface if a cross-model tail-control claim is required.
