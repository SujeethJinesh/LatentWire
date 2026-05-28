# DriftRot DeepSeek Tight-Clip Check

Date: 2026-05-28T23:50Z

## Decision

`AMBIGUOUS_TAIL_BOUND_IMPROVES_MEDIAN_REGRESSES`

Do not claim tight clip `[0.5, 2.0]` as a cross-model universal DriftRot
retune.

## Evidence

Run:
`experimental/outlier_migrate/phase9/results/om_driftrot_deepseek_clip_tight_20260528T2318Z`

Comparison source:
`artifacts/scale_cvar_clip/deepseek_tight_clip_comparison.csv`

| Metric | Baseline ParoQuant | Tight clip |
|---|---:|---:|
| Median recovery | 0.756 | 0.518 |
| CI95 | [-0.246, 0.855] | [0.265, 0.797] |
| Worst trace | -3.056 | -3.524 |
| Mean recovery | 0.176 | 0.073 |

Median per-trace margin: `-0.232`, CI95 `[-0.591, 0.228]`.

## Interpretation

The same tight clip that robustifies Granite does not cleanly transfer to
DeepSeek. It improves the bootstrap lower bound but loses median recovery and
worsens the worst trace. This supports the regime-aware framing: clip/CVaR
settings need model-specific selection rather than a universal replacement for
the ParoQuant baseline.

## Consequence

Keep Granite tight clip as Tier-2 tail-control evidence. Do not promote it to
headline method. A future cross-model version needs a calibration selector or a
different tail objective.
