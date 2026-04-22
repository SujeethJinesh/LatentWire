# Quotient Locality Transport Follow-Up References

Date: `2026-04-22`

## Why This Memo Exists

Pure geometry tweaking looks mostly saturated, but symmetry-aware and
locality-preserving transport are not fully dead yet. If we revisit the math
side, it should be through quotienting and interpretable transport diagnostics,
not another fixed rotation wrapper.

## Strongest References

- [Complete Characterization of Gauge Symmetries in Transformer Architectures](https://openreview.net/forum?id=KrkbYbK0cH)
- [FuLA](https://arxiv.org/abs/2505.20142)
- [I-FuLA](https://openreview.net/forum?id=hJvcbkf2nO)
- [Latent Functional Maps](https://arxiv.org/abs/2406.14183)
- [When Embeddings Models Meet](https://openreview.net/forum?id=DLEzSo1DIk)
- [On the Spectral Geometry of Cross-Modal Representations](https://arxiv.org/abs/2604.08579)
- [Invariance Through Latent Alignment](https://arxiv.org/abs/2112.08526)
- [Unification of Symmetries Inside Neural Networks](https://arxiv.org/abs/2402.02362)

## Sharp Falsification Ablation

1. Quotient out head-wise gauge freedom first.
2. Fit a locality-preserving or functional transport map second.
3. Test on the larger frozen same-pair slice and one matched cross-family pair.

If that still cannot beat the live dynalign residual row, the geometry route is
effectively exhausted for the main paper.

## Interpretable Diagnostics

- map orthogonality / commutativity error
- neighborhood preservation score
- per-layer correspondence stability
- source / target / communicated / oracle on the same IDs
