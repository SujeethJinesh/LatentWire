# Asymmetric K/V Budget Toy Summary

This artifact summarizes the quantization-inspired asymmetric-K/V toy branch.
The method uses one query projection but separates the route budget from the
value budget, then logs how much the two selected slot sets disagree.

## Matched total budget 4

| Scenario | Top-k | Query-pool | Route atom | Constrained precond. | Asym K/V 1+3 | Asym K/V 2+2 |
|---|---:|---:|---:|---:|---:|---:|
| aligned | 0.2396 | 0.3177 | 0.3281 | 0.3594 | 0.4844 | 0.4167 |
| rotated | 0.2760 | 0.2969 | 0.3073 | 0.3906 | 0.4219 | 0.4688 |
| outlier | 0.2448 | 0.3594 | 0.3125 | 0.2812 | 0.4219 | 0.4167 |
| slot_permuted | 0.2604 | 0.3281 | 0.3542 | 0.3177 | 0.4427 | 0.3906 |

## Larger route/value budget 2+4

| Scenario | Asym K/V 2+4 | Rec MSE | KV overlap | KV Jaccard | KV KL | KV cosine | Gate mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| aligned | 0.4063 | 0.8645 | 0.1536 | 0.0646 | 16.0778 | 0.0797 | 0.0439 |
| rotated | 0.3906 | 0.8155 | 0.1589 | 0.0651 | 16.0436 | 0.0693 | 0.0454 |
| outlier | 0.4792 | 0.8358 | 0.1328 | 0.0542 | 16.3027 | 0.0215 | 0.0514 |
| slot_permuted | 0.4010 | 0.8801 | 0.1120 | 0.0464 | 16.4232 | 0.0571 | 0.0537 |

## Interpretation

- `1+3` is the best matched-budget split for aligned, outlier, and
  slot-permuted stress cases.
- `2+2` is the best matched-budget split for rotated stress.
- Both matched-budget asymmetric splits beat the older top-k, query-pool,
  route-atom, and constrained-preconditioning controls in every stress case in
  this toy suite.
- Low route/value overlap and Jaccard, high route/value KL, and low route/value
  cosine show that K-like routing and V-like content preservation are selecting
  different slots. That is the useful mechanism, not just extra capacity.

## Next real-model ablation

Move this from toy space into the real evaluator as an asymmetric K/V budget
control: separate K-route and V-value retention ratios, log separate K/V
distortion or attention-fidelity metrics, and compare against equal-byte
uniform/shuffled controls before stacking it with route-atom head selection.
