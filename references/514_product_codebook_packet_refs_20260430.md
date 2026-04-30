# Product-Codebook Packet References

Date: 2026-04-30

Blocker addressed: the paper needs a compression-native source-private packet
method that is less hand-coded than semantic anchors and more informative than
random sign sketches.

## Sources

1. Product Quantization for Nearest Neighbor Search. IEEE TPAMI, 2011.
   DOI:10.1109/TPAMI.2010.57. https://doi.org/10.1109/TPAMI.2010.57
   - Mechanism: split vectors into subspaces and transmit compact learned
     codebook indices.
   - Experiment implication: product-codebook packets should send one learned
     centroid index per byte and decode against target-side candidate vectors.
   - Role: baseline and direct method inspiration.

2. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
   arXiv:2504.19874. https://arxiv.org/abs/2504.19874
   - Mechanism: rotation and quantization for low-distortion vector transport,
     with PQ-style comparisons.
   - Experiment implication: PQ packets should be compared against scalar WZ,
     QJL residual, protected rotated residual, and sign-only packets on the same
     byte budget and source controls.
   - Role: baseline and systems framing.

3. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
   Overhead. arXiv:2406.03482. https://arxiv.org/abs/2406.03482
   - Mechanism: sign/JL low-bit vector sketching.
   - Experiment implication: the failed rotation-sign gate is the right negative
     control; PQ should beat it if preserving magnitude/codebook structure
     matters.
   - Role: ablation and theory support.

4. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.
   arXiv:2402.02750. https://arxiv.org/abs/2402.02750
   - Mechanism: low-bit KV-cache quantization.
   - Experiment implication: report PQ packets separately from KV compression:
     PQ sends explicit source-private code indices, while KIVI compresses
     cache state.
   - Role: systems baseline.

5. Cache-to-Cache: Direct Semantic Communication Between Large Language Models.
   arXiv:2510.03215. https://arxiv.org/abs/2510.03215
   - Mechanism: source-to-target cache projection/fusion.
   - Experiment implication: C2C remains the closest model-to-model competitor
     if we claim latent/cache communication, but product-codebook packets are
     lower-rate and expose no source KV/cache.
   - Role: competitor and novelty threat.

6. LMCache: Serving LLMs with Distributed KV Cache. arXiv:2510.09665.
   https://arxiv.org/abs/2510.09665
   - Mechanism: KV cache storage/reuse across engines and requests.
   - Experiment implication: useful systems baseline for cache movement and
     offload, but not a source-private evidence-packet method.
   - Role: systems framing.

7. Large-Language-Model Enabled Semantic Communication Systems.
   arXiv:2407.14112. https://arxiv.org/abs/2407.14112
   - Mechanism: LLM-assisted semantic communication over channels.
   - Experiment implication: cite for semantic communication context, while
     emphasizing LatentWire's task-causal source-destroying controls.
   - Role: framing.

8. Diffusion Transformers with Representation Autoencoders.
   arXiv:2510.11690. https://arxiv.org/abs/2510.11690
   - Mechanism: semantically rich latent spaces for diffusion transformers.
   - Experiment implication: motivates later learned latent receivers, but also
     raises the review bar if we overclaim latent transfer from current packet
     results.
   - Role: inspiration and overclaim guard.

## Effect On This Cycle

The product-codebook gate should be treated as a new compression-native method
candidate rather than a pure negative. It functionally passes the remapped
source-control gate, but the current Python decoder fails the strict latency
bar. The next experiment should either optimize/cached-decode the PQ receiver
or scale the functional PQ rows with paired uncertainty while clearly separating
systems-latency claims.
