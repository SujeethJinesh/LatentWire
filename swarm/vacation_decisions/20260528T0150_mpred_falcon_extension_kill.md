# 2026-05-28T01:50Z M-PRED Falcon extension result

## Result

Run directory: `experimental/outlier_migrate/phase9/results/om_phase9_mpred_falcon_extension_20260528T011658Z`

Checker decision: `KILL_MPRED`

| Regime | Median recovery | CI95 |
|---|---:|---:|
| M11b top-5 | 0.070 | [-0.077, 0.101] |
| M11b top-10 | 0.044 | [-0.144, 0.203] |
| static top-10 | -0.025 | [-0.126, 0.170] |
| M-PRED top-10 alpha=0.95 | -0.0049 | [-0.233, 0.202] |

M-PRED trails the best M11b arm by -0.075 median recovery and does not clear
the architecture-fill threshold.

## Decision

M-PRED is now uniformly killed on all tested non-Nemotron surfaces:
Granite, DeepSeek, and Falcon. No Nemotron M-PRED run is justified before a
stronger reason appears, because the predictor failed where it was expected to
help most.

## Interpretation

The predictor occasionally helps individual traces but does not improve median
recovery. The failure pattern is consistent with broadband channel dynamics:
innovation correction adds variance faster than it removes EMA lag.

## Next gate

Proceed to M-FISH if budget permits. If M-FISH also kills, the queued Tier 1.5
architecture-grounded methods become eligible.
