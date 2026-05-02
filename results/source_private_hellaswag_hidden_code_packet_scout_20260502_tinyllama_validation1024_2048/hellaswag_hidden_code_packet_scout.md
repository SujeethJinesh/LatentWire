# HellaSwag Hidden-Code Packet Scout

- pass gate: `False`
- eval slice: `1024:2048`
- default encoder: `hidden_pca4_kmeans64`
- default accuracy: `0.500000`
- packet-only accuracy: `0.501953`
- default delta vs packet-only: `-0.001953`
- default CI95 low vs packet-only: `-0.007812`
- best scout accuracy: `0.507812`
- best scout delta vs packet-only: `0.005859`
- packet: `1B` raw / `4B` framed

## Interpretation

This scout tests the next live branch after source-score codes failed: a source-hidden PCA/k-means discrete code plus a supervised train-only hidden reliability quantizer under the same one-byte packet contract. A pass would justify materializing full TinyLlama validation hidden caches; a fail means simple hidden-state codebooks are not enough and the next branch needs a stronger learned quantizer/crosscoder objective.
