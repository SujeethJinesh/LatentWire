# Product-Codebook n500 Sprint References

Date: 2026-04-30

## Primary Sources And Role

- Product Quantization for Nearest Neighbor Search:
  https://ieeexplore.ieee.org/document/5432202/
  - Role: direct method inspiration and baseline.
  - Use in paper: product-codebook packets are a source-private codec; each
    byte indexes a learned centroid in a subspace.

- Optimized Product Quantization:
  https://openaccess.thecvf.com/content_cvpr_2013/html/Ge_Optimized_Product_Quantization_2013_CVPR_paper.html
  - Role: stronger PQ baseline and geometry ablation.
  - Use in paper: reviewers will expect OPQ/protected-basis variants before
    claiming PQ geometry novelty.

- TurboQuant:
  https://arxiv.org/abs/2504.19874
  - Role: quantization/systems inspiration for mixed scalar + residual packets.
  - Use in paper: motivates TurboResidual framing, QJL residual comparators,
    and near-optimal low-bit vector quantization baselines.

- RaBitQ:
  https://arxiv.org/abs/2405.12497
  - Role: randomized binary quantization baseline.
  - Use in paper: same-byte sign/rotation residual packets must be reported as
    adjacent baselines.

- QJL:
  https://arxiv.org/abs/2406.03482
  - Role: one-bit JL/sign residual baseline and KV inner-product estimator.
  - Use in paper: required low-bit residual/KV compression comparator.

- KIVI:
  https://arxiv.org/abs/2402.02750
  - Role: KV-cache quantization baseline.
  - Use in paper: systems section should compare source-boundary packets with
    KV compression floors and avoid claiming native KV-cache superiority.

- KVQuant:
  https://arxiv.org/abs/2401.18079
  - Role: KV-cache quantization baseline.
  - Use in paper: "why not compress KV?" reviewer comparison.

- C2C:
  https://arxiv.org/abs/2510.03215
  - Role: direct model-to-model KV/latent communication competitor.
  - Use in paper: our distinction is privacy/rate/control rigor, not raw
    high-bandwidth latent sharing.

- KVComm:
  https://arxiv.org/abs/2510.03346
  - Role: selective KV sharing competitor.
  - Use in paper: compare boundary exposure and bytes, not just task accuracy.

- Slepian-Wolf coding:
  https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources
  - Role: theory framing.
  - Use in paper: target and source observations are correlated; source sends a
    residual code decoded with target side information.

- Flow Matching:
  https://arxiv.org/abs/2210.02747
  - Role: future receiver inspiration.
  - Use in paper: a one-step flow/denoising receiver is a future branch, not a
    current contribution.

- Consistency Models:
  https://arxiv.org/abs/2303.01469
  - Role: future one-step denoising receiver inspiration.
  - Use in paper: motivates but does not replace the current PQ gate.

- Scalable Diffusion Models with Transformers:
  https://arxiv.org/abs/2212.09748
  - Role: broader diffusion-transformer inspiration.
  - Use in paper: low priority until we build an actual denoising receiver.

## Reviewer Baselines

- target-only and target-wrapper/no-source
- zero, shuffled, random same-byte, answer-only, answer-masked, public-only,
  target-derived, feature/code permutation, top-codeword knockout
- matched-byte structured text and full structured oracle
- scalar Wyner-Ziv, PQ, OPQ, RaBitQ/QJL, protected residual
- KV compression floors: KIVI, KVQuant, QJL
- direct KV/latent communication: C2C, KVComm

## Current Claim Boundary

The n500 product-codebook packet is a compression-native source-private codec
with cached target-side decode. It does not yet prove protocol-free latent
reasoning or native serving speedup.
