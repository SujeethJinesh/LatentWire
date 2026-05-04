# HellaSwag Decision-Sparse Common-Basis Hidden Packet Gate

- pass gate: `False`
- eval slice: `1024:2048`
- default encoder: `cca_pca64_d8_sae64_top2_dw0p2_l10p001`
- default accuracy: `0.503906`
- packet-only accuracy: `0.501953`
- compact common-basis decoder accuracy: `0.503906`
- Qwen-side-only common-basis accuracy: `0.464844`
- default delta vs packet-only: `0.001953`
- default CI95 low vs packet-only: `-0.001953`
- top-atom knockout accuracy: `0.499023`
- top1/top2 ambiguity-code accuracy: `0.500977`
- top1/top2 ambiguity-code delta vs packet-only: `-0.000977`
- top1/top2 ambiguity-code CI95 low: `-0.002930`
- top1/top2 ambiguity-code no-atom accuracy: `0.501953`
- top1/top2 ambiguity-code best destructive: `target_derived_source_pair_ambiguity_control` (`0.501953`)
- best scout accuracy: `0.503906`
- best scout delta vs packet-only: `0.001953`
- packet: `1B` raw / `4B` framed

## Interpretation

This gate tests the highest-priority learned branch after unsupervised hidden-code and linear crosscoder codebooks failed: train a sparse SAE-like basis in linear CCA/common coordinates with a decision loss, transmit only a one-byte atom-plus-candidate packet, and require the atom to matter under atom shuffle, wrong-row source, and top-atom knockout controls. A pass would promote the common-basis packet branch; a fail weakens shallow SAE/linear common-basis methods and points to nonlinear resamplers or less saturated benchmarks.
