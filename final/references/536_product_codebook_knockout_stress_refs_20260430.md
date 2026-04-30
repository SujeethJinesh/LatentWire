# Product-Codebook Knockout Stress References

Date: 2026-04-30

## Primary Sources And Role

- Product Quantization for Nearest Neighbor Search:
  https://ieeexplore.ieee.org/document/5432202/
  - Role: direct precedent for decomposing a vector into subspaces and sending
    one codeword index per subspace.
  - Use in paper: establishes that the packet is a PQ-style codec, so novelty
    must come from source-private side-information evaluation, not PQ itself.

- Optimized Product Quantization:
  https://openaccess.thecvf.com/content_cvpr_2013/html/Ge_Optimized_Product_Quantization_2013_CVPR_paper.html
  - Role: expected stronger geometry baseline.
  - Use in paper: motivates the next OPQ/protected-basis gate after the public
    mean knockout failed.

- Vector Quantization and Signal Compression:
  https://link.springer.com/book/10.1007/978-1-4615-3626-0
  - Role: canonical vector-quantization/source-coding reference.
  - Use in paper: frames codeword knockout as a codec sensitivity diagnostic.

- Additive Quantization for Extreme Vector Compression:
  https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Babenko_Additive_Quantization_for_2014_CVPR_paper.html
  - Role: multi-codebook compression competitor beyond plain PQ.
  - Use in paper: future stronger quantization baseline if PQ is challenged as
    too standard.

- Composite Quantization:
  https://proceedings.mlr.press/v32/zhangd14.html
  - Role: compact multi-codebook vector compression baseline.
  - Use in paper: additional comparator for a broader compression-native row.

- TurboQuant:
  https://arxiv.org/abs/2504.19874
  - Role: recent rate-distortion and online vector-quantization baseline.
  - Use in paper: systems-facing comparison for low-rate vector-coded packets.

- Slepian-Wolf coding:
  https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources
  - Role: source coding with decoder side information.
  - Use in paper: the target has public side information, while the source sends
    a short private residual code.

- Deep Learning Enabled Semantic Communication Systems:
  https://arxiv.org/abs/2006.10685
  - Role: task-oriented semantic communication precedent.
  - Use in paper: helps position this as task-relevant communication rather than
    exact reconstruction of the source state.

## Reviewer Stress Controls

- Top-margin codeword knockout: damage the packet byte that contributes most to
  the gold-vs-nearest-wrong candidate margin.
- Public-mean replacement: replace the selected byte with a train-public mean
  code rather than an oracle adversarial code.
- Random-subspace replacement: measure whether top-margin selection matters more
  than generic byte damage.
- Payload entropy/collision analysis: report whether packets behave like compact
  example IDs.
- OPQ/protected-basis follow-up: test whether a rotated/protected basis reduces
  lookup-like uniqueness while preserving source-causal lift.

## Claim Boundary

This stress gate supports a diagnostic claim that the n500 PQ row is
margin-sensitive to source-selected codewords. It does not prove a fully
interpretable public-neutral packet, and it does not make PQ itself novel.
