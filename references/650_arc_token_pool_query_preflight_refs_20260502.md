# ARC Token-Pool Query Preflight References

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop is plausible; ICLR full paper remains
  blocked.
- Current story: fixed-byte source-private packets, residual/public-basis
  gates, and systems byte/exposure accounting.
- Exact gap: the token-pool query connector did not produce source-necessary
  target-loss gains.

## Primary Sources

1. Prefix-Tuning: Optimizing Continuous Prompts for Generation.
   <https://arxiv.org/abs/2101.00190>
   - Boundary: virtual prefix tokens are target-side adaptation; LatentWire
     must prove per-example source-conditioned communication.

2. Perceiver IO: A General Architecture for Structured Inputs and Outputs.
   <https://arxiv.org/abs/2107.14795>
   - Boundary: learned query bottlenecks motivate token-pool connectors, but
     the paper-facing claim needs source-necessity controls.

3. Set Transformer: A Framework for Attention-based Permutation-Invariant
   Neural Networks.
   <https://arxiv.org/abs/1810.00825>
   - Boundary: attention pooling over sets is an architectural prior for
     candidate/token pools, not a communication result.

4. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image
   Encoders and Large Language Models.
   <https://arxiv.org/abs/2301.12597>
   - Boundary: Q-Former-style bridges justify frozen-source/frozen-target
     connector design, but multimodal adaptation is not cross-LLM messaging.

5. LLM Augmented LLMs: Expanding Capabilities through Composition.
   <https://arxiv.org/abs/2401.02412>
   - Boundary: CALM composes frozen LMs with cross-attention; LatentWire's
     differentiator must be low-rate source-private packets and controls.

6. Cache-to-Cache: Direct Semantic Communication Between Large Language
   Models.
   <https://openreview.net/forum?id=LeatkxrBCi>
   - Boundary: C2C projects/fuses KV caches. Our systems comparison should
     emphasize smaller packet rate, no raw KV transfer, and source-destroying
     controls.

7. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing.
   <https://openreview.net/forum?id=F7rUng23nw>
   - Boundary: KVComm selectively shares KV pairs. It is a direct systems
     baseline, not the same as fixed-byte source-private packets.

8. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
   <https://arxiv.org/abs/2504.19874>
   - Boundary: TurboQuant is an inner-product preserving KV/vector
     quantization method. It motivates packet quantization and byte accounting,
     but not a learned communication protocol by itself.

9. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.
   <https://arxiv.org/abs/2402.02750>
   - Boundary: KIVI is a KV-cache compression baseline. It sharpens systems
     comparisons around cache-shaped transfer versus compact packets.

10. Scalable Diffusion Models with Transformers.
    <https://arxiv.org/abs/2212.09748>
    - Boundary: DiT supports iterative latent refinement as future method
      inspiration; this preflight is still a one-step connector.

11. Sparse Autoencoders Find Highly Interpretable Features in Language Models.
    <https://arxiv.org/abs/2309.08600>
    - Boundary: SAEs can help define interpretable/common bases, but they do
      not prove cross-model transfer without target-task gains.

12. Relative Representations Enable Zero-shot Latent Space Communication.
    <https://openreview.net/forum?id=SrC-nwieGJ>
    - Boundary: relative/anchor representations motivate row-centered common
      coordinates. The current token-pool result shows that richer hidden
      exposure alone is not enough.

## Reviewer Positioning

The token-pool query connector is close to prefix/query-bottleneck prior art,
so it should not be claimed as novel unless it beats source-destroying
controls. The negative n8 result is useful because it prevents overclaiming:
the paper should say that all-choice token exposure is insufficient on the
current Mac-local target-loss surface, and that the live branch has moved to
candidate-level residual pooling with score/hidden factor ablations.
