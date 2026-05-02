# HellaSwag Receiver Acceptance Gate

- pass gate: `False`
- scout pass gate: `False`
- predeclared default pass gate: `False`
- best scout row: `benefit_ridge` / `hybrid_vote_on_score_agreement_prediction` / `score_only`
- best scout eval accuracy: `0.672217`
- best scout delta vs packet-only: `0.000841`
- default eval accuracy: `0.645234`
- default delta vs packet-only: `-0.000876`

## Lay Explanation

This experiment asks whether a receiver can learn when to ignore the TinyLlama packet and use Qwen's own candidate instead. It trains only on an early prefix, chooses the override rule on a prefix dev split, and scores the frozen rule on later heldout rows. The ridge row is a learned error predictor; the relative-kNN row is a nearest-anchor common-basis test.

## Interpretation

This gate tests the most direct receiver-improvement path after the oracle headroom card. A pass would promote a train-only receiver that beats packet-only. A fail weakens simple selective prediction and relative-neighbor receiver families, pushing the next branch toward richer common-basis supervision, official-train receiver calibration, or a learned query-bottleneck receiver rather than more confidence thresholds.
