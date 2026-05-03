# HellaSwag Fixed Hybrid Option-Position Audit

- pass gate: `True`
- eval rows: `10042`
- candidate-only accuracy: `0.526688`
- fixed hybrid accuracy: `0.532464`
- overall delta vs candidate-only: `0.005776`
- overall CI95 low: `0.002888`
- answer-balanced delta: `0.005779`
- answer-balanced CI95 low: `0.002984`
- positive answer-position count: `4` / `4`
- max fixed-hybrid prediction shift from answer distribution: `0.013045`
- best non-identity global packet permutation accuracy: `0.348636`
- best rowwise derangement accuracy: `0.162517`
- max equivariance sanity diff: `0.000000000000`

## Interpretation

The fixed hybrid packet improvement is not concentrated in one answer slot: it is positive for all four gold answer positions and remains positive under answer-position-balanced resampling. Direct cyclic rolls, non-identity global label remaps, and rowwise derangements of the emitted packet collapse, which weakens a simple slot-prior explanation. Same-permutation equivariance checks catch audit implementation errors. This cached audit cannot prove invariance to true candidate-text permutations because the source model was not rerun under reordered answer options.

## Lay Explanation

We checked whether the tiny hybrid hint only helps when the correct answer is, for example, choice A or choice B. It helps a little for every answer position, and if we rotate the hint to the wrong option number the accuracy collapses. That makes the result less likely to be just an answer-position trick, though a future stronger test should rerun the model with the choices physically shuffled.
