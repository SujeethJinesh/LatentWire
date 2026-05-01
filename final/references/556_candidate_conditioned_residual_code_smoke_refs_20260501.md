# Candidate-Conditioned Residual Code Smoke References, 2026-05-01

## Purpose

This memo supports the learned candidate-conditioned residual-code smoke. The
local result prunes a learned receiver-side calibration layer over the existing
candidate-local residual score surface.

Local artifact:
`results/source_private_candidate_conditioned_residual_code_smoke_20260501/`

Code:
`scripts/run_source_private_candidate_conditioned_residual_code_smoke.py`

## Primary Sources and Boundaries

- Product Quantization for Nearest Neighbor Search. IEEE TPAMI 2011.
  https://ieeexplore.ieee.org/document/5432202/
  - Boundary: product quantization is prior work for compact vector codes.
    LatentWire should not claim novelty for PQ/RVQ-style residual coding.
- Additive Quantization for Extreme Vector Compression. CVPR 2014.
  https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Babenko_Additive_Quantization_for_2014_CVPR_paper.html
  - Boundary: additive residual codebooks are prior art; the local novelty must
    be candidate-conditioned side-information communication.
- EnCodec: High Fidelity Neural Audio Compression. arXiv:2210.13438.
  https://arxiv.org/abs/2210.13438
  - Boundary: residual vector quantization is mature in neural codecs. It is an
    inspiration, not the contribution.
- QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
  Overhead. arXiv:2406.03482. https://arxiv.org/abs/2406.03482
  - Boundary: one-bit random residual/sketch correction is prior work.
- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
  arXiv:2504.19874. https://arxiv.org/abs/2504.19874
  - Boundary: TurboQuant already uses vector quantization plus QJL-style
    residual correction for KV/cache inner-product preservation.
- QuIP: 2-Bit Quantization of Large Language Models With Guarantees. NeurIPS
  2023. https://arxiv.org/abs/2307.13304
  - Boundary: incoherence/rotation-assisted quantization is prior work for
    making low-bit codes behave better.
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice
  Codebooks. ICML 2024. https://proceedings.mlr.press/v235/tseng24a.html
  - Boundary: structured rotations and lattice codebooks are prior art for
    compression geometry.
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound.
  SIGMOD 2024. https://arxiv.org/abs/2405.12497
  - Boundary: high-dimensional vector quantization with error accounting is a
    compression baseline, not source-private communication.
- Denoising Diffusion Probabilistic Models. NeurIPS 2020.
  https://arxiv.org/abs/2006.11239
  - Boundary: denoising/refinement is prior art; LatentWire should not claim
    novelty for iterative correction alone.
- Diffusion Posterior Sampling for General Noisy Inverse Problems. ICLR 2023.
  https://openreview.net/forum?id=OnD9zGAGT0k
  - Boundary: posterior refinement using a prior plus measurement likelihood
    motivates candidate-posterior refinement, but is not the current method.
- Consistency Models. ICML 2023. https://arxiv.org/abs/2303.01469
  - Boundary: one/few-step refinement is prior art and should only motivate a
    future receiver refiner.
- Slepian-Wolf Coding. IEEE TIT 1973.
  https://www.mit.edu/~6.454/www_fall_2001/kusuma/slepwolf.pdf
  - Boundary: decoder-side information coding is the right theory frame.
- Wyner-Ziv Coding. IEEE TIT 1976.
  https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf
  - Boundary: the receiver's candidate set should be framed as side
    information; the packet is not self-contained.
- Distributed Source Coding Using Syndromes (DISCUS). IEEE TIT 2003.
  https://www.researchgate.net/publication/2352091_Distributed_Source_Coding_Using_Syndromes_DISCUS_Design_and_Construction
  - Boundary: syndrome/binning language is prior art. The local claim must be
    empirical LLM candidate-posterior disambiguation under strict controls.

## Local Finding

The learned receiver-side calibration layer fails the gate. It keeps controls
near target, but matched accuracy collapses below the base residual receiver:

- core-to-holdout: `0.625` learned versus `0.875` base;
- holdout-to-core: `0.250` learned versus `0.750` base;
- same-family-all: `0.500` learned versus `0.812` base.

The control-weight sweep shows the tradeoff: lower control weight recovers more
matched signal but leaks controls; higher control weight preserves controls but
falls back to the prior.

## Novelty Boundary

The safe future claim is:

> learned Wyner-Ziv-style latent communication where the receiver's candidate
> posterior is decoder-only side information and a tiny residual/syndrome
> disambiguates that posterior across models.

The failed local branch is not that method yet. It learns a receiver-side rule
over a fixed packet, so it should be pruned rather than promoted.
