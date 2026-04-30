# Product-Codebook Geometry Knockout References

Date: 2026-04-30

## Primary Sources And Role

- Product Quantization for Nearest Neighbor Search:
  https://doi.org/10.1109/TPAMI.2010.57
  - Role: base PQ method.
  - Use in paper: canonical PQ is the adjacent codec baseline; novelty is the
    source-private decoder-side-information evaluation and geometry stress.

- Optimized Product Quantization:
  https://openaccess.thecvf.com/content_cvpr_2013/html/Ge_Optimized_Product_Quantization_2013_CVPR_paper.html
  - Role: OPQ rotation baseline.
  - Use in paper: utility-initialized OPQ is the public-mean knockout-sensitive
    geometry variant.

- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice
  Codebooks:
  https://arxiv.org/abs/2402.04396
  - Role: Hadamard incoherence and lattice/codebook quantization precedent.
  - Use in paper: motivates protected Hadamard geometry as a hardware-friendly
    low-bit rotation rather than a dense learned transform.

- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs:
  https://openreview.net/forum?id=dfqsW38v1X
  - Role: structured rotations for low-bit inference.
  - Use in paper: supports the systems claim that sign/permutation/Hadamard
    rotations are plausible for accelerator-friendly packet codecs.

- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate:
  https://arxiv.org/abs/2504.19874
  - Role: recent vector-quantization systems baseline.
  - Use in paper: positions geometry-mitigated PQ against modern
    rate-distortion quantization rather than treating PQ as new.

- Slepian-Wolf coding:
  https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources
  - Role: decoder-side-information theory.
  - Use in paper: the target has public candidate state; the source sends a
    residual code over private evidence.

- Wyner-Ziv coding:
  https://ieeexplore.ieee.org/document/1054651
  - Role: lossy source coding with decoder side information.
  - Use in paper: frames scalar WZ and PQ packets as lossy residual codes
    decoded with target-side public state.

## Reviewer Controls Motivated

- Public-mean top-codeword knockout: tests whether useful evidence survives a
  source-neutral replacement.
- Payload entropy and collision-conditioned accuracy: tests whether accuracy is
  only carried by singleton packet IDs.
- Basis-seed stability: tests whether the result is robust to structured
  rotations rather than one lucky codebook.
- Structured-rotation systems accounting: sign/permutation/Hadamard transforms
  should be reported separately from dense OPQ rotations.

## Claim Boundary

The n500 gate supports geometry-mitigated source-private PQ as a method
contribution. It does not make PQ or OPQ novel by themselves, and it does not
claim general cross-model latent reasoning.
