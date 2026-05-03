# HellaSwag Fixed Hybrid Full-Validation Gate

- pass gate: `True`
- eval rows: `10042`
- candidate-only accuracy: `0.526688`
- fixed hybrid accuracy: `0.532464`
- hybrid delta vs candidate-only: `0.005776`
- hybrid CI95 low vs candidate-only: `0.002888`
- positive slice count: `10` / `10`
- candidate/hybrid oracle accuracy: `0.540530`

## Interpretation

The fixed hybrid vote-on-score-agreement packet extends from the prior strict 0:9216 surface to the full cached HellaSwag validation range 0:10042, including the previously unresolved terminal tail. This strengthens the packet-policy evidence and evaluation-quality story, but it remains a fixed-byte candidate-id packet rather than a learned common latent receiver.

## Lay Explanation

We checked the last cached HellaSwag examples that were not part of the previous large strict surface. The same tiny hybrid answer hint still helps on that tail and on the full validation set, so the current packet result is not just from the first 9216 examples.
