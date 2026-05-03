# HellaSwag Disagreement-Prototype Receiver Gate

- pass gate: `False`
- scout pass gate: `False`
- default pass gate: `False`
- packet-only accuracy: `0.619199`
- default receiver accuracy: `0.619299`
- default delta vs packet-only: `0.000100`
- best scout receiver accuracy: `0.620394`
- best scout delta vs packet-only: `0.001195`
- best destructive/control delta vs packet-only: `-0.000498`
- target-or-packet oracle accuracy: `0.686815`

## Lay Explanation

This gate looks for reusable disagreement shapes. On official train rows, it finds cases where Qwen's alternative answer would fix the TinyLlama packet and cases where Qwen would hurt it. Those groups become small prototypes. On validation, the receiver only overrides the TinyLlama packet when the current row looks closer to the helpful prototypes than to the harmful prototypes.

## Interpretation

This directly tests the next receiver/common-basis hypothesis from the ledger. A pass would mean that official-train disagreement prototypes can capture the TinyLlama/Qwen oracle headroom without validation-label tuning. A fail means the current HellaSwag receiver gap is not solved by shallow prototype geometry; the next live branch should move to an actual query-bottleneck/nonlinear connector or to a stronger source/benchmark surface.
