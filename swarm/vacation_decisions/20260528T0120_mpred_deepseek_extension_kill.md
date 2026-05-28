# 2026-05-28T01:20Z M-PRED DeepSeek extension result

## Result

Run directory: `experimental/outlier_migrate/phase9/results/om_phase9_mpred_deepseek_extension_20260528T005045Z`

Checker decision: `KILL_MPRED`

| Regime | Median recovery | CI95 |
|---|---:|---:|
| M11b top-10 | 0.335 | [-0.409, 0.494] |
| static top-10 | 0.377 | [-1.142, 0.538] |
| M-PRED top-10 alpha=0.95 | -0.043 | [-1.145, 0.402] |

M-PRED trails M11b top-10 by -0.379 median recovery and does not clear the
architecture-fill threshold.

## Interpretation

DeepSeek weakens M-PRED further: the high-alpha predictive arm neither improves
on M11b nor beats static top-10. This supports the Granite diagnosis that the
innovation term is reacting to noisy channel changes rather than predicting a
stable next protected set.

## Next gate

Run the same extension on Falcon-H1. Falcon is still informative because M11b
was weakest there and the throughput is expected to fit within the remaining
budget.
