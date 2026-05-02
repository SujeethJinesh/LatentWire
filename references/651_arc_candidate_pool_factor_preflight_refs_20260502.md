# ARC Candidate-Pool Factor Preflight References

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop is plausible; ICLR full paper remains
  blocked.
- Current story: fixed-byte source-private packets, residual/public-basis
  gates, and systems byte/exposure accounting.
- Exact gap: candidate-level score/hidden query pooling did not produce a
  source-necessary target-loss gain.

## Primary Sources

1. Cache-to-Cache: Direct Semantic Communication Between Large Language
   Models.
   <https://openreview.net/forum?id=LeatkxrBCi>
   - Boundary: C2C owns learned KV-cache projection/fusion. LatentWire must
     distinguish compact source-private packets from cache-shaped transfer.

2. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing.
   <https://openreview.net/forum?id=F7rUng23nw>
   - Boundary: selective KV sharing is a direct inter-LLM systems baseline and
     motivates layer/head importance controls.

3. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
   <https://openreview.net/forum?id=tO3ASKZlok>
   - Boundary: TurboQuant is a strong rate-distortion comparator for vector/KV
     compression, not evidence of source-necessary communication.

4. KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache
   Quantization.
   <https://arxiv.org/abs/2401.18079>
   - Boundary: per-channel/pre-RoPE/outlier-aware KV quantization sets a
     serious compressed-KV baseline for systems claims.

5. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.
   <https://arxiv.org/abs/2402.02750>
   - Boundary: KIVI is a simple recognizable KV compression baseline that any
     packet byte frontier should compare against.

6. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large
   Language Models.
   <https://arxiv.org/abs/2306.14048>
   - Boundary: attention-score heavy hitters motivate score-only controls; a
     score factor alone is not a communication method.

7. SnapKV: LLM Knows What You are Looking for Before Generation.
   <https://arxiv.org/abs/2404.14469>
   - Boundary: query/observation-window based KV selection is a systems
     neighbor for deciding which source states are worth transmitting.

8. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image
   Encoders and Large Language Models.
   <https://arxiv.org/abs/2301.12597>
   - Boundary: Q-Former justifies learned query bottlenecks, but not the claim
     that our candidate pool carries source-necessary information.

9. Flamingo: A Visual Language Model for Few-Shot Learning.
   <https://arxiv.org/abs/2204.14198>
   - Boundary: Perceiver-style resampling into frozen models is architectural
     precedent, not novelty by itself.

10. Slepian-Wolf: Noiseless Coding of Correlated Information Sources.
    <https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources>
    - Boundary: target-side candidate state is decoder side information; the
      next packet should encode conditional source residuals, not marginal
      hidden state.

11. Wyner-Ziv: The Rate-Distortion Function for Source Coding with Side
    Information at the Decoder.
    <https://cir.nii.ac.jp/crid/1360564063947537280>
    - Boundary: motivates a conditional packet/syndrome view where the target
      already knows the public question and candidate text.

12. Prefix-Tuning: Optimizing Continuous Prompts for Generation.
    <https://arxiv.org/abs/2101.00190>
    - Boundary: soft prefixes are a known frozen-LM adaptation interface. The
      new contribution must be source-conditioned and source-necessary.

## Reviewer Positioning

The candidate-pool factor gate separates three possible sources of apparent
gain: cached source selected-choice signal, source hidden candidate residuals,
and their concatenation. None passes the n8 source-necessity gate. This is
useful negative evidence: the next method should not be another shallow
soft-prefix pooling variant. It should encode a target-conditional candidate
innovation packet and report a rate-distortion curve against compressed-KV and
KV-sharing baselines.
