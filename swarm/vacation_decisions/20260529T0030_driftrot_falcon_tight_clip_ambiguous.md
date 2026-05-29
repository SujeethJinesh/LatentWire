# Decision: Falcon Tight Clip Does Not Promote Universal DriftRot Retune

Timestamp: 2026-05-29T00:30Z

## Run

- Experiment: DriftRot Scale/CVaR/Clip cross-model check on Falcon-H1
- Run directory: `experimental/outlier_migrate/phase9/results/om_driftrot_falcon_clip_tight_20260528T2343Z`
- Comparison table: `artifacts/scale_cvar_clip/falcon_tight_clip_comparison.csv`
- Config: ParoQuant-style rotation with scale clip `[0.5, 2.0]`, reusing the same BF16/static/protected-set sources as the Falcon ParoQuant baseline.

## Result

| Metric | Baseline Falcon ParoQuant | Tight clip `[0.5, 2.0]` |
|---|---:|---:|
| Median recovery | 0.381 | 0.390 |
| CI95 | [0.0645, 0.547] | [0.0019, 0.637] |
| Worst trace | -1.323 | -1.610 |
| Mean recovery | 0.217 | 0.213 |

The generic Falcon checker still reports `PASS_V1_PAROQUANT_FALCON_ROTATION_RESCUE` because tight clip remains well above Falcon M11b. That is not the DriftRot decision surface. Against Falcon's own ParoQuant baseline, tight clip is flat-to-worse.

## Decision

`AMBIGUOUS_MEDIAN_FLAT_WORST_REGRESSES`.

Tight clip remains valid Granite-specific tail-control evidence, but it does
not become a cross-model ParoQuant retune. The current safe claim is:
Granite has a recoverable tail that tight clipping fixes; DeepSeek and Falcon
do not support the same clip as a universal setting.

## Next Gate

Do not spend more GPU on this exact clip setting. The next positive-method gate
should either add a model-specific selector explaining when tight clipping is
appropriate, or pivot to a surface/branch diagnostic that changes the protected
surface rather than only retuning ParoQuant clipping.
