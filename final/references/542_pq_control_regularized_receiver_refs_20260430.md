# PQ Control-Regularized Receiver References

Date: 2026-04-30

## Primary Sources And Role

- Product Quantization for Nearest Neighbor Search:
  https://doi.org/10.1109/TPAMI.2010.57
  - Role: product-codebook primitive used by the packet.
  - Use in paper: adjacent codec primitive; not a novelty claim.

- QJL:
  https://arxiv.org/abs/2406.03482
  - Role: compact residual/sign sketching baseline.
  - Use in paper: next residual-packet inspiration after the disjoint PQ
    collapse.

- TurboQuant:
  https://arxiv.org/abs/2504.19874
  - Role: rotation, quantization, and residual correction precedent.
  - Use in paper: motivates TurboResidual PQ/QJL packets, but LatentWire must
    claim source-private communication controls rather than new quantization.

- DDPM:
  https://arxiv.org/abs/2006.11239
  - Role: denoising from corrupted state.
  - Use in paper: analogy only; packet corruption views can be trained toward
    clean decisions.

- D3PM:
  https://arxiv.org/abs/2107.03006
  - Role: discrete denoising diffusion.
  - Use in paper: analogy for discrete packet corruption and recovery.

- MaskGIT:
  https://arxiv.org/abs/2202.04200
  - Role: masked discrete token reconstruction.
  - Use in paper: analogy for masked packet consistency, not a generative
    image-model claim.

- Consistency Models:
  https://proceedings.mlr.press/v202/song23a.html
  - Role: one-step mapping from corrupted/noisy state to a consistent endpoint.
  - Use in paper: motivates receiver objectives that map clean/masked packets
    to gold and destroyed packets to target prior.

- Slepian-Wolf coding:
  https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources
  - Role: source coding with correlated decoder information.
  - Use in paper: target public candidate state is decoder side information.

- Wyner-Ziv coding:
  https://ieeexplore.ieee.org/document/1055508
  - Role: lossy coding with decoder side information.
  - Use in paper: mathematical framing for source-private residual packets.

- Distributed indirect source coding with decoder side information:
  https://arxiv.org/abs/2405.13483
  - Role: modern side-information reference.
  - Use in paper: supports the conditional innovation framing.

- BLIP-2 / Q-Former:
  https://arxiv.org/abs/2301.12597
  - Role: learned query bottleneck connector between frozen systems.
  - Use in paper: future learned connector inspiration, not a current result.

- Flamingo:
  https://arxiv.org/abs/2204.14198
  - Role: Perceiver-style resampling into a language model.
  - Use in paper: supports the interface-redesign branch if PQ remains
    disjoint-unsafe.

- Relative Representations:
  https://arxiv.org/abs/2209.15430
  - Role: common-basis/anchor-relative representation precedent.
  - Use in paper: supports the idea that shared bases matter; LatentWire's
    current contribution is packet/control evaluation, not representation
    theory.

- Cache-to-Cache:
  https://arxiv.org/abs/2510.03215
  - Role: closest high-rate cache communication competitor.
  - Use in paper: boundary reference; LatentWire must not claim broad
    cache-to-cache superiority without native baselines.

- KVCOMM:
  https://arxiv.org/abs/2510.12872
  - Role: online KV-cache communication/reuse competitor.
  - Use in paper: systems comparator and non-claim boundary.

## Novelty Boundary

Novel:

- Controls are part of the learned receiver objective: source-destroying,
  random, permuted, and deranged-public-table packets are trained toward the
  target prior.
- The learned receiver is evaluated as a source-private packet consumer with
  exact byte and exposure accounting inherited from the PQ systems waterfall.
- The result explicitly reports the disjoint-ID collapse rather than hiding it.

Not novel:

- PQ, residual sketching, denoising/consistency learning, query bottlenecks,
  or side-information source-coding theory.
- Broad latent communication or KV/cache sharing.

## Claims Enabled

- A learned linear score adapter can preserve the established
  utility-protected-Hadamard PQ signal on exact-overlap n500 surfaces while
  keeping deranged public-table and source-destroying controls near target.
- The same branch fails on disjoint IDs because the underlying deterministic PQ
  signal collapses. This is now the main blocker for an ICLR headline.

## Non-Claims

- No disjoint-safe learned receiver has been shown.
- No protocol-free latent reasoning.
- No native GPU/vLLM serving result.
- No claim that PQ/OPQ/Hadamard/TurboQuant/QJL are new primitives.
