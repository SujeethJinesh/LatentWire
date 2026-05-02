# HellaSwag Disagreement-Prototype Receiver

- pass gate: `False`
- predeclared default pass gate: `False`
- scout pass gate: `False`
- control separation gate: `False`
- official train calibration rows: `1487`
- validation rows: `10042`
- default eval accuracy: `0.618602`
- default delta vs packet-only: `-0.000597`
- best scout eval accuracy: `0.620693`
- best scout delta vs packet-only: `0.001494`
- full-validation Tiny+Qwen oracle: `0.686815`

## Lay Explanation

The scalar receiver asked whether the packet or Qwen looked globally more confident. This experiment instead builds train-only prototypes of rows where TinyLlama and Qwen disagree. At validation time, it asks whether a row is near past disagreement types where Qwen helped more than it harmed. Random, label-permuted, and score-row-shuffled controls test whether any lift comes from real local disagreement structure.

## Interpretation

This is a common-basis probe, not a learned soft-prompt or KV-cache fusion method. If it passes, it promotes local disagreement prototypes as a receiver mechanism. If it fails, the evidence says the remaining Tiny/Qwen oracle headroom likely needs a richer sparse/crosscoder or learned query-bottleneck receiver rather than local prototype thresholds.
