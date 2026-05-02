# HellaSwag Sparse-Query Receiver

- pass gate: `False`
- predeclared default pass gate: `False`
- scout pass gate: `False`
- control separation gate: `False`
- official train calibration rows: `1487`
- validation rows: `10042`
- default eval accuracy: `0.617407`
- default delta vs packet-only: `-0.001792`
- best scout eval accuracy: `0.619797`
- best scout delta vs packet-only: `0.000597`
- full-validation Tiny+Qwen oracle: `0.686815`

## Lay Explanation

Previous receivers only looked at score summaries or local prototypes. This experiment looks at the full hidden-state difference between Qwen's candidate and the TinyLlama packet candidate, compresses that large residual into a tiny set of train-only query features, and learns when those features say Qwen should override the packet.

## Interpretation

This is a sparse-query common-basis probe, not C2C, prefix tuning, or KV-cache transfer. A pass would promote full candidate residual queries as the missing receiver mechanism. A failure would mean that even a low-rank view of the full Qwen candidate residuals cannot close the Tiny/Qwen oracle on the current official-train calibration surface.
