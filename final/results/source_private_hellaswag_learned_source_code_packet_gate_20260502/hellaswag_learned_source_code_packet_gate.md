# HellaSwag Learned Source-Code Packet Gate

- pass gate: `False`
- default encoder: `packet_z_quantile_32`
- default accuracy: `0.615316`
- packet-only accuracy: `0.619199`
- default delta vs packet-only: `-0.003884`
- default CI95 low vs packet-only: `-0.006326`
- best scout accuracy: `0.619896`
- best scout delta vs packet-only: `0.000697`
- packet: `1B` raw / `4B` framed

## Interpretation

This gate tests whether changing the source encoder to a train-only learned discrete code can recover the Tiny/Qwen oracle headroom after receiver-only and residual-logit branches saturated. A pass would promote a learned source-code packet under the compact one-byte contract; a fail means source-score-derived discrete codes also do not beat packet-only on the current HellaSwag calibration surface.
