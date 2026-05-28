# 2026-05-28T00:50Z M-PRED Granite reduced result

## Result

The reduced Granite M-PRED run completed the two theory-matched treatment arms and was intentionally early-stopped before random controls to preserve GPU budget.

Run directory: `experimental/outlier_migrate/phase9/results/om_phase9_mpred_granite_reduced_20260527T201251Z`

Checker decision: `KILL_MPRED`

Completed regimes:

- `mpred_top10_alpha_0_95`
- `mpred_top5_alpha_0_95`

Key medians on the 8 positive-gap traces:

| Regime | Median recovery | CI95 |
|---|---:|---:|
| M11b top-5 | 0.449 | [-1.301, 1.001] |
| M11b top-10 | 0.241 | [-0.727, 0.436] |
| M-PRED top-10 alpha=0.95 | -7.746 | [-389.686, 0.319] |
| M-PRED top-5 alpha=0.95 | -1.960 | [-12.616, 0.050] |

The best M-PRED arm trails the best M11b arm by -2.410 median recovery.

## Decision

Close Granite M-PRED as a KILL. The control arms were skipped after both treatment arms lost decisively to M11b; a random-alpha control cannot rescue a treatment-vs-baseline failure this large.

## Interpretation

The one-step innovation term appears to overreact to noisy channel-magnitude changes. This matches the FFT diagnostic: the channel spectrum is broadband with high entropy, so EMA lag is not the only limiting factor. The result weakens predictive-channel tracking as the next positive-method branch.

## Next gate

Per the queued user clarification, extend M-PRED to DeepSeek and Falcon only if the existing V2 packets can be reused safely and the checker can be extended without changing the Granite decision. These are the architectures where M11b was ambiguous or failed, so they are the remaining informative M-PRED surfaces before moving to M-FISH.
