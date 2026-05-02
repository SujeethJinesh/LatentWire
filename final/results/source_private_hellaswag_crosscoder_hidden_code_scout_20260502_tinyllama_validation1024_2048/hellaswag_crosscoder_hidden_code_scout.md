# HellaSwag Crosscoder Hidden-Code Scout

- pass gate: `False`
- eval slice: `1024:2048`
- default encoder: `cca_pca64_d8_relconf_q32_ridge10`
- default accuracy: `0.503906`
- packet-only accuracy: `0.501953`
- compact crosscoder decoder accuracy: `0.501953`
- default delta vs packet-only: `0.001953`
- default delta vs compact crosscoder: `0.001953`
- best scout accuracy: `0.507812`
- best scout delta vs packet-only: `0.005859`
- packet: `1B` raw / `4B` framed

## Interpretation

This scout tests a train-only linear crosscoder/CCA-style shared basis after raw-hidden and anchor-relative codebooks failed. A pass would revive the learned common-basis branch; a fail means linear shared projections are also insufficient and the next branch should use a nonlinear resampler/cross-attention connector or a less packet-saturated benchmark.
