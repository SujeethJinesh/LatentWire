# Toy Gauge-Fix Quotient Bridge Sweep

- seed: `7`
- heads: `4`
- head dim: `6`
- classes: `5`
- families: `5`
- held-out family: `4`
- anchor family: `0`
- seen shots / class: `20`
- held-out shot grid: `[1, 2, 4]`

This toy isolates whether gauge fixing and quotient-aware head matching help held-out-family transport when the observed source heads differ by family-specific permutations and per-head linear gauges.

References:
- Complete Characterization of Gauge Symmetries in Transformer Architectures: https://openreview.net/forum?id=KrkbYbK0cH
- GaugeKV: https://openreview.net/forum?id=rSxYPLzyBu

| Shot | Method | Accuracy | MSE | dAcc vs few-shot | dMSE vs few-shot | Centroid cosine | Gauge residual | Head-match acc | Shared Basis |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1.0000 | heldout_fewshot_ridge | 1.0000 | 0.0985 | 0.0000 | 0.0000 | 0.9843 | 0.0443 | - | False |
| 1.0000 | global_seen_ridge | 0.5750 | 1.7099 | -0.4250 | 1.6114 | 0.5807 | 1.5211 | - | False |
| 1.0000 | gauge_fix_then_bridge | 1.0000 | 0.1665 | 0.0000 | 0.0680 | 0.9826 | 0.2808 | 0.0000 | True |
| 1.0000 | quotient_match_after_fix | 1.0000 | 0.0796 | 0.0000 | -0.0189 | 0.9880 | 0.0696 | 1.0000 | True |
| 1.0000 | oracle_family_ridge | 1.0000 | 0.0012 | 0.0000 | -0.0973 | 1.0000 | 0.0019 | - | False |
| 2.0000 | heldout_fewshot_ridge | 1.0000 | 0.0410 | 0.0000 | 0.0000 | 0.9958 | 0.0136 | - | False |
| 2.0000 | global_seen_ridge | 0.5750 | 1.7099 | -0.4250 | 1.6689 | 0.5807 | 1.5211 | - | False |
| 2.0000 | gauge_fix_then_bridge | 1.0000 | 0.1893 | 0.0000 | 0.1483 | 0.9743 | 0.2814 | 0.0000 | True |
| 2.0000 | quotient_match_after_fix | 1.0000 | 0.0693 | 0.0000 | 0.0283 | 0.9901 | 0.0639 | 1.0000 | True |
| 2.0000 | oracle_family_ridge | 1.0000 | 0.0012 | 0.0000 | -0.0398 | 1.0000 | 0.0019 | - | False |
| 4.0000 | heldout_fewshot_ridge | 1.0000 | 0.0141 | 0.0000 | 0.0000 | 0.9989 | 0.0047 | - | False |
| 4.0000 | global_seen_ridge | 0.5750 | 1.7099 | -0.4250 | 1.6958 | 0.5807 | 1.5211 | - | False |
| 4.0000 | gauge_fix_then_bridge | 1.0000 | 0.1467 | 0.0000 | 0.1326 | 0.9810 | 0.2491 | 0.0000 | True |
| 4.0000 | quotient_match_after_fix | 1.0000 | 0.0505 | 0.0000 | 0.0364 | 0.9932 | 0.0401 | 1.0000 | True |
| 4.0000 | oracle_family_ridge | 1.0000 | 0.0012 | 0.0000 | -0.0129 | 1.0000 | 0.0019 | - | False |

## Best Non-Oracle by Shot
- 1 shot/class: best accuracy `quotient_match_after_fix` (1.0000 acc, 0.0796 MSE); lowest non-oracle MSE `quotient_match_after_fix` (0.0796); best shared-basis `quotient_match_after_fix` (0.0796 MSE)
- 2 shot/class: best accuracy `heldout_fewshot_ridge` (1.0000 acc, 0.0410 MSE); lowest non-oracle MSE `heldout_fewshot_ridge` (0.0410); best shared-basis `quotient_match_after_fix` (0.0693 MSE)
- 4 shot/class: best accuracy `heldout_fewshot_ridge` (1.0000 acc, 0.0141 MSE); lowest non-oracle MSE `heldout_fewshot_ridge` (0.0141); best shared-basis `quotient_match_after_fix` (0.0505 MSE)
