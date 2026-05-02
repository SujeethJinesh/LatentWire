# References: HellaSwag Anchor-Relative Hidden-Code Scout

## Claim Boundary

This memo supports the failed anchor-relative hidden-code scout. The safe claim
is that train-only anchor-relative source-hidden byte codes were evaluated on a
frozen HellaSwag slice and did not robustly beat compact packet-only. It does
not claim a new anchor representation, a universal latent language, SAE or
crosscoder alignment, prefix-token transfer, KV/cache communication, or native
serving speedup.

## Primary Sources And Why They Matter

1. Relative Representations
   - https://openreview.net/forum?id=SrC-nwieGJ
   - Boundary: similarity-to-anchor representations for zero-shot latent-space
     communication are direct prior art. Our novelty cannot be the anchor
     coordinate system itself.

2. Locally Linear Embedding
   - https://www.science.org/doi/10.1126/science.290.5500.2323
   - Boundary: preserving local neighborhood geometry is classic manifold
     learning. Local anchor behavior is not new.

3. Laplacian Eigenmaps
   - https://papers.nips.cc/paper/1961-laplacian-eigenmaps-and-spectral-techniques-for-embedding-and-clustering
   - Boundary: graph/local spectral bases are established dimensionality
     reduction tools and should be treated as motivation, not novelty.

4. Random Fourier Features
   - https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines
   - Boundary: random/common kernel bases are prior art. A random or Fourier
     coordinate frame alone would not be a unique contribution.

5. Feature Hashing
   - https://icml.cc/Conferences/2009/papers/407.pdf
   - Boundary: compact shared hashed coordinates are prior art. The fixed-byte
     packet contract and source-destroying controls are the relevant novelty.

6. Sparse autoencoder feature-space universality
   - https://arxiv.org/abs/2410.06981
   - Boundary: shared sparse feature spaces across LLMs are prior art and a
     future comparator. This scout does not train an SAE.

7. Universal Sparse Autoencoders
   - https://arxiv.org/abs/2502.03714
   - Boundary: USAE-style shared concept spaces are stronger common-basis
     methods than this shallow anchor-relative codebook.

8. Sparse Crosscoders
   - https://transformer-circuits.pub/2024/crosscoders/index.html
   - Boundary: learned cross-model dictionaries remain the next higher-value
     branch. This artifact only tests shallow relative-coordinate codes.

9. Prefix-Tuning and P-Tuning v2
   - https://aclanthology.org/2021.acl-long.353/
   - https://arxiv.org/abs/2110.07602
   - Boundary: prefix/prompt tuning sends or learns continuous virtual-token
     conditioning. LatentWire packets are per-example discrete byte records and
     are not consumed as soft prompts.

10. C2C, KVComm, and KVCOMM
    - https://openreview.net/forum?id=LeatkxrBCi
    - https://arxiv.org/abs/2510.03346
    - https://arxiv.org/abs/2510.12872
    - Boundary: cache/KV communication is a different high-rate systems
      interface. This scout transmits no source KV, raw hidden, or raw scores.

11. QJL, TurboQuant, KIVI, and KVQuant
    - https://arxiv.org/abs/2406.03482
    - https://arxiv.org/abs/2504.19874
    - https://arxiv.org/abs/2402.02750
    - https://arxiv.org/abs/2401.18079
    - Boundary: these are vector/KV compression methods. Our artifact is a
      task-level byte packet and cannot claim vector-fidelity or HBM wins.

12. Diffusion and Diffusion Transformers
    - https://arxiv.org/abs/2006.11239
    - https://arxiv.org/abs/2212.09748
    - Boundary: iterative denoising/refinement remains a possible future
      receiver. No diffusion-style latent refinement is implemented here.

## Reviewer-Facing Framing

Safe:

- We tested the most direct shallow anchor/common-basis alternative after raw
  source-hidden codebooks failed.
- One anchor seed nearly reached the scout threshold, but CI crossed zero and
  adjacent anchor-seed repeats collapsed to `+0.001` to `+0.002`.
- The result narrows the next method branch toward joint crosscoder/resampler
  objectives rather than more analytic common-basis codebooks.

Unsafe:

- Claiming anchor-relative representation is novel.
- Claiming a shared latent language.
- Claiming equivalence to SAE/crosscoder alignment.
- Claiming superiority over prefix tokens, C2C/KVComm, or KV quantization.
- Claiming GPU serving or HBM wins before native NVIDIA measurements.
