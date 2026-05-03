# HellaSwag Qwen Hybrid-To-Phi Conditional Acceptor Gate

- pass gate: `False`
- eval rows: `768`
- selected rule: `selected_margin <= 0.213526913511`
- fixed hybrid accuracy: `0.467448`
- conditional acceptor accuracy: `0.454427`
- delta vs fixed hybrid: `-0.013021`
- CI95 low vs fixed hybrid: `-0.028646`
- target-or-hybrid oracle accuracy: `0.604167`

## Interpretation

The conditional acceptor tests whether Phi's own score simplex can safely override the fixed Qwen hybrid packet. This is the cached target-aware receiver branch recommended after shallow source-side vetoes failed. A failure means the current target-score acceptor cannot access the large target-or-hybrid oracle headroom without sacrificing packet utility.

## Lay Explanation

Phi sometimes has useful information that could improve the Qwen hint. This test learns a tiny rule for when Phi should override the hint. On new rows, that rule must beat simply trusting the Qwen hybrid hint.
