# HellaSwag Anchor-Relative Hidden-Code Scout

- pass gate: `False`
- eval slice: `1024:2048`
- default encoder: `anchor64_relpca4_kmeans4`
- default accuracy: `0.502930`
- packet-only accuracy: `0.501953`
- compact relative decoder accuracy: `0.501953`
- default delta vs packet-only: `0.000977`
- default delta vs compact relative decoder: `0.000977`
- best scout accuracy: `0.503906`
- best scout delta vs packet-only: `0.001953`
- packet: `1B` raw / `4B` framed

## Interpretation

This scout tests whether anchor-relative local coordinates are a better common basis than raw source-hidden PCA/k-means codes. A pass would revive the common-basis branch and justify full-validation materialization; a fail means shallow anchor-relative codebooks are also insufficient and the next branch should use a true joint crosscoder/resampler objective or a less packet-saturated benchmark.
