# HellaSwag PQ Hidden Innovation Codec Gate

- pass gate: `False`
- eval slice: `1024:2048`
- default encoder: `pq_pca16_m4_k2_identity`
- default accuracy: `0.497070`
- packet-only accuracy: `0.501953`
- default delta vs packet-only: `-0.004883`
- default CI95 low vs packet-only: `-0.017578`
- best scout accuracy: `0.508789`
- best scout delta vs packet-only: `0.006836`
- packet: `1B` raw / `4B` framed

## Lay Explanation

The experiment asks whether TinyLlama can send more than just its answer choice without sending a hidden vector. It compresses the hidden-state residual into several tiny subcodes, packs those subcodes plus the answer id into one byte, and lets Qwen decode it with its own scores.

## Interpretation

This is the cached Mac-local product-quantization branch suggested by the hidden-code and TurboQuant/PQ literature. A pass would justify materializing more TinyLlama hidden validation caches. A failure means factorized source-hidden codebooks still do not add stable task information beyond the compact candidate packet on this HellaSwag slice.
