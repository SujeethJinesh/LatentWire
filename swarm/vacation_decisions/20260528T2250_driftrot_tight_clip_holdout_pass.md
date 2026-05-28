# DriftRot Tight-Clip Held-Out Pass

Date: 2026-05-28T22:50Z

## Decision

`PROMOTE_TIGHT_CLIP_FINAL_POSITIVE_TRACE`

Run the final remaining positive-gap Granite trace (`prompt_index=8`) with
tight clip `[0.5, 2.0]` before turning this into a paper table.

## Evidence

Run:
`experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_holdout_20260528T2144Z`

Held-out prompts `[1, 2, 5]` were not in the inspected tight-clip screen
`[4, 7, 9, 10]`.

| Prompt | Baseline ParoQuant | Tight clip | Margin |
|---:|---:|---:|---:|
| 1 | 1.567 | 1.647 | +0.081 |
| 2 | 0.477 | 0.847 | +0.370 |
| 5 | 0.997 | 0.996 | -0.001 |

Held-out median recovery: `0.996`. CI95: `[0.847, 1.647]`.

## Interpretation

This supports tight clip as a Tier-2 DriftRot tail-control candidate: it fixes
the inspected Granite tail and does not regress the held-out positive-gap
subset. It is not framed as the headline method, because config/clip retuning
has higher novelty risk than residual correction, surface selection, or
branch-local rotation.

## Next Gate

Score prompt 8 with tight clip to complete all eight recoverable Granite traces.
Then decide whether to present a compact Granite positive-trace table or run
another model/tail surface.
