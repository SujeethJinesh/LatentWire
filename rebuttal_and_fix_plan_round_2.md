# Rebuttal and Fix Plan Round 2

Date: 2026-05-29

Round 2 found no new workshop-level blockers. Remaining concerns are mostly honest limitations that require new experiments or implementation work, which are outside this final polish loop.

## Issues and Disposition

| Concern | Class | Action |
|---|---|---|
| Novelty versus ParoQuant | HONEST-LIMITATION | Keep baseline-not-our-method language. Do not claim a new rotation quantizer. Rebuttal frames the contribution as long-decode drift measurement, failure taxonomy, and a descriptive checklist. |
| No measured systems performance | HONEST-LIMITATION | No kernel/profiler claims are added. Systems-cost table remains an analytical envelope for future residual selectors after K-RES failed. |
| Small N and wide CIs | HONEST-LIMITATION | Keep negative CIs and included-trace counts visible. Do not strengthen inferential language. |
| BCa/Holm auditability | FIXABLE-IN-PROSE | Added a Reproducibility Statement sentence pointing to `e15_bca_holm.json`, which records packet, regime, raw p value, Holm rank, adjusted p value, threshold, and reject flag for all 33 tests. |
| Protocol not held-out validated | HONEST-LIMITATION | Already stated in Section 4.5; keep "provisionally suggests" and "confirmation packet" language. |
| MATH-500 overread | HONEST-LIMITATION + EXISTING FIX | Existing text says MATH-500 supports drift generalization, not recovery generalization. No further scope change. |
| Local ParoQuant-style fidelity | HONEST-LIMITATION | Existing caveat states it is a local implementation following reported scaled pairwise-rotation design, not official code. |

## Anticipated Reviewer Responses

**No new quantizer.** Correct. The paper argues that a strong prior-work rotation baseline absorbs the recoverable signal in these packets, while original-basis channel-set methods and residual selectors do not add reliably. That is a mechanism finding and a design warning.

**Analytical systems numbers.** Correct. The paper treats the cost model as an envelope, not a measured implementation result. It is included to make the failed residual-correction branch concrete for systems readers.

**Small-N ratios.** Correct. The paper reports the uncertainty visibly and uses the results to support scoped mechanism claims rather than a universal deployment rule.

**Protocol validation.** Correct. The checklist is prospective in design and descriptive in this paper's evaluation. Held-out validation remains future work.

## Status

One safe prose fix was applied. No numerical claim, scope, limitation, or baseline ownership changed.

