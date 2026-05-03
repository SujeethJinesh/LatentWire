# HellaSwag Qwen Hybrid-To-Phi Cross-Family Gate

- pass gate: `True`
- heldout eval rows: `768`
- Phi target-only accuracy: `0.263021`
- Qwen candidate-only packet accuracy: `0.455729`
- Qwen hybrid packet accuracy: `0.467448`
- hybrid delta vs candidate-only: `0.011719`
- hybrid CI95 low vs candidate-only: `0.001302`
- target-or-hybrid oracle accuracy: `0.604167`

## Interpretation

The fixed Qwen hybrid vote-on-score-agreement packet improves over candidate-only on the cached Phi cross-family heldout rows while retaining the same receiver-visible one-candidate packet contract. This is useful cross-family packet-policy survival, but it is not a learned Phi receiver or a general latent language: the receiver still sees only the final candidate id.

## Lay Explanation

We checked whether the improved Qwen hint still helps when the receiving model is Phi instead of Qwen. On the cached Phi rows, the fixed hybrid hint beats both Phi's own answer and the older Qwen candidate-only hint. The caveat is that Phi is still just receiving an answer-choice hint, not a rich hidden thought.
