# HellaSwag Conditional Selector/Syndrome Gate

- pass gate: `False`
- default source code: `packet_z_q32`
- default alternative: `qwen_target_score`
- default accuracy: `0.618901`
- packet-only accuracy: `0.619199`
- default delta vs packet-only: `-0.000299`
- default CI95 low vs packet-only: `-0.000896`
- best scout accuracy: `0.621291`
- best scout delta vs packet-only: `0.002091`
- packet: `1B` raw / `4B` framed

## Lay Explanation

This experiment trains a tiny referee. The source still sends only a compact byte packet. When the source packet and Qwen disagree, the referee predicts whether switching to Qwen's candidate is likely to help or harm, using only official-train calibration labels.

## Interpretation

This is the first post-headroom method gate aimed directly at the Tiny/Qwen disagreement surface. A pass would promote a source-private conditional selector/syndrome method under a one-byte packet contract. A fail means the measured oracle headroom is not recovered by linear train-only benefit prediction over packet id, source confidence bins, and Qwen score features; the next method would need a nonlinear resampler/cross-attention connector or a different benchmark surface.
