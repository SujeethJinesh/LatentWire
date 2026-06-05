# Mock COLM Review Board: Channel-Set

paper: `paper/channel_set/draft.md`
date: 2026-06-05
verdict: `WEAK_ACCEPT`
ac_decision: `ACCEPT_IF_MEASUREMENT_CLAIM_ONLY`
honesty_trust_gate: `PASS`

## Reviewer A

Overall: `WEAK_ACCEPT`

Scores:

- Quality/soundness: 6
- Significance: 5
- Originality: 5
- Clarity: 6
- Honesty/trust: 7
- Relation to prior work: 5

Strengths: The drift measurement is clear and numerically meaningful. Top-1% strict set-leaving above 0.53 across all reported packets is a useful challenge to static outlier-channel protection.

Weaknesses: The paper currently lacks a confirmed positive method. It should be presented as a measurement/regime paper, not as an adaptive-policy paper.

Required revision: preserve the C-A1 contamination disclosure and keep positive-method language out of the abstract.

## Reviewer B

Overall: `WEAK_ACCEPT`

Scores:

- Quality/soundness: 6
- Significance: 5
- Originality: 5
- Clarity: 6
- Honesty/trust: 7
- Relation to prior work: 5

Strengths: Separating strict set-leaving from within-set shuffling is helpful. The threshold sweep reduces concern that the result is a single chosen threshold.

Weaknesses: The draft needs final figures and a provenance table mapping each packet to hashes/artifacts. ParoQuant parity is correctly described as required before any future positive, but it is not yet shown for C-A1.

Required revision: add the threshold sensitivity plot and an artifact/provenance table before submission.

## Reviewer C

Overall: `WEAK_ACCEPT`

Scores:

- Quality/soundness: 5
- Significance: 5
- Originality: 5
- Clarity: 6
- Honesty/trust: 7
- Relation to prior work: 5

Strengths: The audit discipline is unusually strong. Parking C-A1 after detecting confirm contamination increases trust in the remaining measurement.

Weaknesses: A bounded-negative Channel-Set paper must make a stronger case that static-set drift is independently useful to readers, not only a failed method search.

Required revision: move the drift measurement and regime map to the front of the paper and leave C-A1 as future work.

## Area Chair

Decision: `ACCEPT_IF_MEASUREMENT_CLAIM_ONLY`

Rationale: The board is weak-accept if and only if the paper stays within the measurement/regime claim. The top-1% strict set-leaving evidence is strong enough for a scoped contribution, and the contamination audit prevents an overclaim. A confirmed adaptive method would require fresh non-confirm paired materialization and full baseline gates, which the draft already states.
