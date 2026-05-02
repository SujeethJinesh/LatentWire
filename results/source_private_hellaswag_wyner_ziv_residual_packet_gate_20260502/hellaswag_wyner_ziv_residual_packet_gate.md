# HellaSwag Wyner-Ziv Residual-Logit Packet Gate

- pass gate: `False`
- default accuracy: `0.616013`
- packet-only accuracy: `0.619199`
- default delta vs packet-only: `-0.003187`
- default CI95 low vs packet-only: `-0.004979`
- best prior receiver scout accuracy: `0.620594`
- default delta vs best prior receiver scout: `-0.004581`
- best scout accuracy: `0.619199`
- best scout delta vs packet-only: `0.000000`
- packet: `2B` raw / `5B` framed
- packet bits used by default: `10`

## Interpretation

This gate tests the strongest Mac-feasible conditional-coding branch after receiver selectors saturated. Instead of sending only the TinyLlama candidate id, the packet also carries a tiny quantized TinyLlama score-residual sketch that the decoder combines with Qwen side information. A pass would promote a Wyner-Ziv-style conditional packet; a fail means score-residual source coding does not close the Tiny/Qwen receiver headroom on this calibration surface.
