# HellaSwag Fixed-Hybrid True Candidate-Text Permutation Gate

- pass gate: `True`
- eval rows: `1024`
- original fixed-hybrid accuracy: `0.465820`
- remapped fixed-hybrid accuracy: `0.463867`
- canonical consistency: `0.964844`
- score cache hit: `False`
- hidden cache hit: `False`

## Interpretation

This evaluates whether the full fixed-hybrid prediction row follows candidate text under a fresh hidden-pipeline candidate permutation, rather than only testing cached label remaps. It is still a bounded Mac-local smoke unless widened to larger slices and more permutations.
