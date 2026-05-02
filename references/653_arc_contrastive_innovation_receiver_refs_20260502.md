# ARC Contrastive Innovation Receiver References

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper
  remains blocked.
- Current story: source-private fixed-byte packets are being tested with
  destructive controls and systems byte/exposure accounting.
- Exact gap: the source-control contrastive innovation receiver does not yet
  beat candidate-roll and target-cache controls.

## Primary Sources

1. van den Oord et al., "Representation Learning with Contrastive Predictive
   Coding." <https://arxiv.org/abs/1807.03748>
   - Boundary: InfoNCE/contrastive learning is prior art. Our novelty cannot
     be contrastive loss; it must be source-necessity evidence under
     corrupted-packet controls.

2. Khosla et al., "Supervised Contrastive Learning."
   <https://arxiv.org/abs/2004.11362>
   - Boundary: supervised contrastive objectives are standard representation
     learning tools. LatentWire's claim must be the communication protocol and
     falsification ladder.

3. Burges et al., "Learning to Rank using Gradient Descent."
   <https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/>
   - Boundary: pairwise ranking losses are known. The ARC gate uses a capped
     margin penalty as an engineering tool, not a new ranking method.

4. Robinson et al., "Contrastive Learning with Hard Negative Samples."
   <https://openreview.net/forum?id=CR1XOQ0UTh->
   - Boundary: hard negatives motivate zero/shuffle/noise/candidate-roll
     controls, but the paper contribution must be matched-source falsification
     for cross-model packets.

5. Slepian and Wolf, "Noiseless Coding of Correlated Information Sources."
   <https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources>
   - Boundary: side-information coding motivates conditional packets, but it
     does not establish heterogeneous LLM latent communication.

6. Wyner and Ziv, "The Rate-Distortion Function for Source Coding with Side
   Information at the Decoder."
   <https://ieeexplore.ieee.org/document/1055039>
   - Boundary: the public-innovation residual is a decoder-side-information
     analogy, not a claim of optimal source coding.

7. Li and Liang, "Prefix-Tuning: Optimizing Continuous Prompts for
   Generation." <https://arxiv.org/abs/2101.00190>
   - Boundary: soft prefixes are known frozen-LM adapters. A LatentWire packet
     is per-example source-conditioned side information; the gate fails unless
     matched source beats target-cache and corrupted-source controls.

8. Fu et al., "Cache-to-Cache: Direct Semantic Communication Between Large
   Language Models." <https://arxiv.org/abs/2510.03215>
   - Boundary: C2C projects and fuses source KV caches. LatentWire should be
     compared as an extreme-rate packet interface that does not transmit raw
     KV/hidden state.

9. Shi et al., "KVComm: Enabling Efficient LLM Communication through Selective
   KV Sharing." <https://arxiv.org/abs/2510.03346>
   - Boundary: selective KV sharing is a direct systems competitor; our
     systems claim must be byte/exposure regime separation until native serving
     rows exist.

10. Du et al., "Enabling Agents to Communicate Entirely in Latent Space."
    <https://arxiv.org/abs/2511.09149>
    - Boundary: latent-agent communication is adjacent prior art. LatentWire's
      distinctive claim must be answer-key-forbidden fixed-byte packets and
      destructive same-row controls.

11. Zandieh et al., "TurboQuant: Online Vector Quantization with
    Near-optimal Distortion Rate." <https://arxiv.org/abs/2504.19874>
    - Boundary: TurboQuant is a vector/KV compression comparator. It can
      challenge byte-only claims, but it is still a source-state codec rather
      than a tiny task-level packet.

12. "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
    Overhead." <https://arxiv.org/abs/2406.03482>
    - Boundary: QJL motivates sign-sketch byte floors for systems comparisons;
      it does not solve source-private conditional reasoning by itself.

13. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache."
    <https://arxiv.org/abs/2402.02750>
    - Boundary: KIVI is a strong KV-cache compression baseline for systems
      byte floors, not a fixed-byte source-private packet method.

14. "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache
    Quantization." <https://arxiv.org/abs/2401.18079>
    - Boundary: KVQuant strengthens the compressed-state baseline. LatentWire
      needs to report task accuracy versus source-state byte floors.

15. Kwon et al., "Efficient Memory Management for Large Language Model Serving
    with PagedAttention." <https://arxiv.org/abs/2309.06180>
    - Boundary: vLLM/PagedAttention defines the serving-memory context.
      LatentWire cannot claim native throughput superiority until measured on
      NVIDIA serving stacks.

## Experiment Implication

The source-control contrastive run narrows the novelty claim. Contrastive
training is an implementation detail; the paper-worthy contribution would be a
source-private packet interface that survives candidate-roll, candidate
derangement, zero-source, shuffle, same-norm noise, and target-cache controls
at a fixed byte budget. The current soft-prefix receiver does not yet meet
that standard.
