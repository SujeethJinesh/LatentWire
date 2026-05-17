# Post-M18 Analysis 2: Per-Position Recovery Curves
## Verdict
Per-position recovery curves are **not identifiable from existing method packets**. M2, M10, M11, and M18 all score a single 512-token window ending at decode position 10000, so the packets support endpoint recovery comparisons but not recovery-vs-position curve shapes.
## Endpoint recovery summary
| Method | Packet | Decision | Primary/baseline regime | Median recovery | CI95 | No-gap fraction | Curve shape support |
|---|---|---|---|---:|---|---:|---|
| M2 position-conditioned sets | `experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z` | `KILL_M2_RANDOM_CONTROL_BEATS` | `primary` | -0.866837313391 | [-3.435289251335, 0.595238665766] | 0.333333333333 | endpoint only; no curve |
| M10 hard-binned scales | `experimental/outlier_migrate/phase9/results/om_phase9_m10_granite_small_vac12_20260515T085800Z` | `KILL_M10_RANDOM_CONTROL_BEATS` | `primary` | 0.234447927834 | [-2.072905653259, 0.503521494450] | 0.000000000000 | endpoint only; no curve |
| M11 EMA smoothing | `experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z` | `KILL_M11_AMBIGUOUS` | `m11_alpha_0_5` | 0.048299284138 | [-8.941156151014, 0.704599255873] | 0.333333333333 | endpoint only; no curve |
| M18 activation+K coupling | `experimental/outlier_migrate/phase9/results/om_phase9_m18_granite_small_vac12_20260516T193500Z` | `KILL_M18_AMBIGUOUS` | `m18_activation_k` | -0.343590844126 | [-14.071191710979, 0.740602124280] | 0.333333333333 | endpoint only; no curve |

## Interpretation
The available endpoint data show that the failed methods fail at the long-decode scoring point, but they cannot distinguish short-decode benefit followed by long-decode decay from uniform failure. A future curve experiment should score identical regimes at multiple endpoint windows, e.g. 2000, 5000, 10000, and 20000, before making claims about decay shape.
