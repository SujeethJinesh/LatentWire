# Rotation-Sign Packet Gate References

Date: 2026-04-30

Blocker addressed: the paper needs a compression-native packet contribution
that is not just a hand-coded semantic-anchor sidecar. The tested gate asks
whether random-rotation sign sketches can carry source-private evidence under
the same destructive controls as the promoted packet rows.

## Sources

1. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
   Overhead. arXiv:2406.03482. https://arxiv.org/abs/2406.03482
   - Mechanism: Johnson-Lindenstrauss projection with sign-bit quantization.
   - Experiment implication: use sign/JL packets as equal-byte source-private
     packet baselines, but require constrained-shuffle, answer-masked,
     permuted-bit, and random controls.
   - Role: baseline, ablation, theory support.

2. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
   arXiv:2504.19874. https://arxiv.org/abs/2504.19874
   - Mechanism: rotation plus quantization for low-distortion vector transport.
   - Experiment implication: keep protected rotated residual packets as the
     fairer TurboQuant/QJL-inspired comparator; do not promote plain sign
     sketches if they lose to scalar Wyner-Ziv packets.
   - Role: baseline and systems framing.

3. Product Quantization for Nearest Neighbor Search. IEEE TPAMI, 2011.
   DOI:10.1109/TPAMI.2010.57. https://doi.org/10.1109/TPAMI.2010.57
   - Mechanism: transmit compact learned codebook indices for approximate
     nearest-neighbor matching.
   - Experiment implication: a product-codebook source packet remains a stronger
     next compression-native branch than uncalibrated random sign sketches.
   - Role: baseline and next-method inspiration.

4. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.
   arXiv:2402.02750. https://arxiv.org/abs/2402.02750
   - Mechanism: asymmetric low-bit KV-cache quantization.
   - Experiment implication: KV/cache compression is a byte-floor comparator,
     not the same mechanism, because it transports/stores source cache state
     rather than a tiny auditable source-private packet.
   - Role: systems baseline.

5. SnapKV: LLM Knows What You are Looking for Before Generation.
   OpenReview. https://openreview.net/forum?id=poE54GOq2l
   - Mechanism: query-aware KV cache pruning/selection.
   - Experiment implication: cache-selection baselines motivate explicit
     target-only and target-derived sidecar controls; packet gains must not be
     target-cache artifacts.
   - Role: systems baseline and ablation pressure.

6. Cache-to-Cache: Direct Semantic Communication Between Large Language Models.
   arXiv:2510.03215. https://arxiv.org/abs/2510.03215
   - Mechanism: project/fuse source cache state into target cache state.
   - Experiment implication: keep C2C as the closest direct-communication
     competitor, but separate it from source-private packet rows by access
     assumption, bytes, and source-state exposure.
   - Role: closest competitor and framing.

7. Deep Learning Enabled Semantic Communication Systems / DeepSC.
   arXiv:2006.10685. https://arxiv.org/abs/2006.10685
   - Mechanism: optimize semantic recovery rather than exact bit recovery.
   - Experiment implication: useful framing for semantic communication, but the
     LatentWire claim should remain task-causal and control-driven, not
     reconstruction-driven.
   - Role: theory and framing.

8. NVIDIA CUDA C++ Best Practices Guide, coalesced global memory access.
   https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
   - Mechanism: memory transaction and coalescing constraints bound realized
     transfer traffic.
   - Experiment implication: report raw payload bytes and rounded 64B/128B
     transfer accounting; do not claim a single request physically moves only 2
     bytes on hardware.
   - Role: systems framing and overclaim guard.

## Uniqueness Read

The closest prior work covers vector compression, KV/cache compression, or
direct cache-to-cache transfer. I did not find a primary-source prior that
combines all of: source-private rate-capped packets, target-side candidate side
information, task-level cross-model decision improvement, and strict
source-destroying controls. The novelty story is therefore strongest as
auditable source-private packet communication, not generic latent compression.

## Effect On Next Experiment

The rotation-sign gate failed and should be treated as a pruned baseline. The
next compression-native experiment should either learn a product-codebook packet
or keep the protected rotated residual family only as a systems comparator
against the promoted scalar/semantic-anchor rows.
