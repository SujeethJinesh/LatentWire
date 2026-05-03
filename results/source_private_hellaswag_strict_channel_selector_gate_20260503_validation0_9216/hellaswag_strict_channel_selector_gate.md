# HellaSwag Strict Channel-Selector Gate

- positive method pass: `True`
- eval rows: `9216`
- candidate-only accuracy: `0.525499`
- best non-oracle method: `fixed_hybrid_vote_on_score_agreement`
- best accuracy: `0.531141`
- best delta vs candidate-only: `0.005642`
- best CI95 low: `0.002713`
- channel oracle selected+vote+trained+score: `0.593099`

## Interpretation

The static hybrid vote-on-score-agreement policy beats the 1B candidate-only packet with positive paired uncertainty on the strict 0:9216 HellaSwag surface. This strengthens the packet-policy contribution, but the train-prefix selectors still do not learn a stronger per-row receiver from this cache. The remaining ICLR blocker is still a learned receiver, common-basis transfer method, cross-family generalization, or native systems evidence.

## Lay Explanation

The source packet machinery produces several possible answer choices. We tested whether a simple rule could choose when to trust the vote channel instead of the default selected answer. The fixed rule helps reliably on this frozen HellaSwag slice, but the learned selectors still did not discover a better general rule.

