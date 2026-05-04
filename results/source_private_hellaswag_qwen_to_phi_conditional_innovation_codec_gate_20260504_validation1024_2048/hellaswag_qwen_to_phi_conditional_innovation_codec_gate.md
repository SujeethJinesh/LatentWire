# HellaSwag Qwen-To-Phi Conditional Innovation Codec Gate

- pass gate: `False`
- calibration rows: `1487`
- eval rows: `768`
- selected rate bytes: `1`
- fixed hybrid accuracy: `0.467448`
- conditional innovation accuracy: `0.467448`
- delta vs fixed hybrid: `0.000000`
- CI95 low vs fixed hybrid: `0.000000`
- ghost-only accuracy: `0.309896`
- best destructive control: `code_value_permutation_innovation_control` at `0.388021`

## Interpretation

This gate tests the conditional innovation hypothesis: if Phi can already predict some of Qwen's candidate frontier from its local scores, the source should spend packet bits only on the residual. A pass would promote residual coding as the next cross-family method branch. A failure means this linear ghost plus discrete residual codec is not sufficient, even though richer residual channels may still be alive.

## Lay Explanation

First we ask Phi to guess what Qwen probably thinks about the four answers. Then Qwen sends only a tiny correction to that guess. The controls replace that correction with fake or target-only versions to check whether the real source correction is doing useful work.
