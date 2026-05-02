# ARC Public-Innovation Preflight References

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop is plausible; ICLR full paper remains
  blocked.
- Current story: fixed-byte source-private packets, public-basis/residual
  diagnostics, destructive controls, and systems byte/exposure accounting.
- Exact gap: the public-side-information innovation soft-prefix gate still
  does not beat target-only and source-destroying controls.

## Primary Sources

1. Slepian and Wolf, "Noiseless Coding of Correlated Information Sources."
   <https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources>
   - Boundary: distributed coding motivates conditional rates, but it does not
     by itself solve heterogeneous LLM latent communication.

2. Wyner and Ziv, "The Rate-Distortion Function for Source Coding with Side
   Information at the Decoder."
   <https://www.itsoc.org/publications/papers/the-rate-distortion-function-for-source-coding-with-side-information-at-the-decoder>
   - Boundary: the public-ridge residual is a Wyner-Ziv-style diagnostic, not a
     new information-theory claim.

3. Cache-to-Cache: Direct Semantic Communication Between Large Language
   Models. <https://arxiv.org/abs/2510.03215>
   - Boundary: C2C moves and fuses projected KV caches. LatentWire must claim a
     much smaller source-private conditional packet, not broad latent
     communication.

4. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing.
   <https://arxiv.org/abs/2510.03346>
   - Boundary: selective KV sharing is the direct systems communication
     baseline; our packet must be compared as a rate/byte/exposure tradeoff.

5. KVCOMM: Online Cross-context KV-cache Communication for Efficient
   LLM-based Multi-agent Systems. <https://arxiv.org/abs/2510.12872>
   - Boundary: online cross-context KV-cache exchange supports the systems
     motivation, but it is not a source-private fixed-byte residual packet.

6. Interlat: Enabling Agents to Communicate Entirely in Latent Space.
   <https://arxiv.org/abs/2511.09149>
   - Boundary: Interlat studies last-hidden-state latent communication between
     agents, including heterogeneous models. LatentWire must distinguish
     answer-key-forbidden conditional packets and destructive controls.

7. Prefix-Tuning: Optimizing Continuous Prompts for Generation.
   <https://arxiv.org/abs/2101.00190>
   - Boundary: soft prefixes are known frozen-LM adapters. Our novelty cannot
     be the soft-prefix interface; it must be per-example source-conditioned
     communication.

8. The Power of Scale for Parameter-Efficient Prompt Tuning.
   <https://arxiv.org/abs/2104.08691>
   - Boundary: prompt tuning learns task prompts. A LatentWire packet must fail
     under zero-source/shuffle controls if it is genuinely communication.

9. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image
   Encoders and Large Language Models. <https://arxiv.org/abs/2301.12597>
   - Boundary: Q-Former motivates lightweight query bottlenecks between frozen
     systems, but not source-private LLM-to-LLM packets.

10. Perceiver IO: A General Architecture for Structured Inputs & Outputs.
    <https://arxiv.org/abs/2107.14795>
    - Boundary: learned latent queries over arbitrary inputs support the query
      connector design; the contribution must be in source-necessity evidence.

11. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
    <https://arxiv.org/abs/2504.19874>
    - Boundary: TurboQuant is a strong vector/KV rate-distortion comparator and
      a source of residual/QJL design ideas, not a cross-model reasoning method.

12. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
    Overhead. <https://arxiv.org/abs/2406.03482>
    - Boundary: QJL motivates sign/sketch residual packets and unbiased
      inner-product preservation, but current public-ridge innovation is dense
      and has not passed source controls.

13. Scalable Diffusion Models with Transformers.
    <https://arxiv.org/abs/2212.09748>
    - Boundary: diffusion-transformer work motivates iterative refinement as a
      lateral design pattern only. It should not be cited as direct prior art
      for model-to-model latent communication.

## Experiment Implication

The public-innovation gate narrows the next branch. A pure train-only public
ridge residual is not enough; the receiver needs an explicit source-control
contrastive objective and packet-level candidate-roll controls before widening
to larger ARC/OpenBookQA slices or NVIDIA systems measurements.
