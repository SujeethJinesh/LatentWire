# COLM_v3 Disagreement Audit

This audit uses the frozen acceptance-baseline predictions and asks where the packet changes the target decision.

| Benchmark | seed-items | target | source-index | packet | source-only correct | target-only correct | packet follows source | repairs target | damages target | repairs source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ARC-Challenge | 11720 | 0.265 | 0.346 | 0.344 | 0.265 | 0.185 | 0.995 | 0.264 | 0.185 | 0.001 |
| OpenBookQA | 10000 | 0.276 | 0.378 | 0.378 | 0.266 | 0.164 | 0.999 | 0.266 | 0.164 | 0.001 |

Interpretation: the packet mostly follows the source-selected candidate. It repairs target errors primarily when the source is correct, and it can damage target-correct/source-wrong rows. This supports the paper's source-choice boundary rather than a stronger evidence-synthesis claim.
