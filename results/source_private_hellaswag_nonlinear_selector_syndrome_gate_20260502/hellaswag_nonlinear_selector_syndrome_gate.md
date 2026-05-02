# HellaSwag Nonlinear Selector/Syndrome Gate

- pass gate: `False`
- default source code: `top2_margin_q16`
- default alternative: `qwen_target_score`
- default RFF: `64` components, gamma `1.0`, seed `19`
- default accuracy: `0.616610`
- packet-only accuracy: `0.619199`
- default delta vs packet-only: `-0.002589`
- default CI95 low vs packet-only: `-0.003884`
- default oracle-headroom capture: `-0.031669`
- best scout accuracy: `0.621390`
- best scout delta vs packet-only: `0.002191`
- packet: `1B` raw / `4B` framed

## Lay Explanation

This experiment asks whether the previous failure was just too linear. TinyLlama still sends only a tiny byte packet. The receiver uses a small nonlinear referee, trained only on official HellaSwag train examples, to decide whether to trust TinyLlama's packet or switch to Qwen's candidate.

## Interpretation

A pass would promote a fixed-byte nonlinear conditional syndrome method. A fail means the HellaSwag oracle headroom is not recovered by a bounded nonlinear train-only receiver over the current packet/syndrome and Qwen score features; the next live method must change the source information itself, move to a true joint connector on GPU, or cut the HellaSwag receiver-improvement claim.
