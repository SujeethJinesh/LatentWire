# HellaSwag Complementarity Headroom Gate

- pass gate: `True`
- eval rows: `10042`
- source packet accuracy: `0.619199`
- target-side accuracy: `0.532464`
- target-or-source oracle accuracy: `0.686815`
- oracle lift vs source: `0.067616`
- oracle CI95 low vs source: `0.062637`
- target-correct/source-wrong rows: `679`
- disagreement rate: `0.292472`
- positive blocks: `5/5`

## Interpretation

This cache-only gate measures whether the existing source packet and target-side prediction have complementary errors. It justifies a conditional syndrome/selector method only if the oracle lift is stable; it does not itself transmit new information.
