# HellaSwag Fixed-Hybrid Candidate-Text Permutation Slice Repeat References

Date: 2026-05-04

## Why This Memo Exists

This memo records the related-work and novelty boundary for the non-overlapping
`1024:2048` HellaSwag physical candidate-text permutation gate. The result is a
robustness repeat for a fixed source-private candidate packet, not a learned
latent-language or native systems result.

## Primary Sources And Boundaries

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - Link: https://arxiv.org/abs/1905.07830
   - Role: benchmark for adversarially filtered commonsense continuation.
   - Boundary: HellaSwag is multiple-choice, so candidate-order and
     candidate-text hardening are required before making communication claims.

2. Large Language Models Are Not Robust Multiple Choice Selectors
   - Link: https://arxiv.org/abs/2309.03882
   - Role: option-ID/selector-bias warning.
   - Boundary: this repeat physically reorders candidate text, but still uses
     one non-identity order per row rather than all `24` orders per row.

3. Large Language Models Sensitivity to The Order of Options in
   Multiple-Choice Questions
   - Link: https://arxiv.org/abs/2308.11483
   - Role: motivates candidate-text reordering rather than cached label remaps.
   - Boundary: the current gate tests stability under a bounded shuffle
     schedule, not universal option-order invariance.

4. Prefix-Tuning: Optimizing Continuous Prompts for Generation
   - Link: https://aclanthology.org/2021.acl-long.353/
   - Role: continuous soft-prefix baseline.
   - Boundary: prefix tuning learns persistent task conditioning; this gate
     evaluates a per-example source-conditioned candidate packet.

5. Learning to Compress Prompts with Gist Tokens
   - Link: https://arxiv.org/abs/2304.08467
   - Role: prompt/context compression baseline.
   - Boundary: gist tokens compress visible context for reuse; this gate
     transmits no source text or reusable source memory object.

6. Training Large Language Models to Reason in a Continuous Latent Space
   - Link: https://arxiv.org/abs/2412.06769
   - Role: latent-reasoning comparator.
   - Boundary: Coconut feeds hidden states back into the same model as
     continuous thoughts; this gate is task-packet transfer under source-private
     constraints.

7. DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM
   Serving
   - Link: https://arxiv.org/abs/2411.02820
   - Role: KV-cache communication and serving comparator.
   - Boundary: DroidSpeak reuses and recomputes KV-cache layers; this gate
     transmits no source KV or dense intermediate state.

8. Enabling Agents to Communicate Entirely in Latent Space
   - Link: https://arxiv.org/abs/2511.09149
   - Role: latent hidden-state communication comparator.
   - Boundary: Interlat transmits last-layer hidden states and compressed
     latent communications; this gate transmits a byte-scale candidate packet.

9. Sparse Crosscoders for Cross-Layer Features and Model Diffing
   - Link: https://transformer-circuits.pub/2024/crosscoders/index.html
   - Role: inspiration for the next shared-feature/common-basis branch.
   - Boundary: crosscoders expose shared/exclusive sparse features; the current
     result is only fixed-policy candidate-packet hardening.

10. Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language
    Models
    - Link: https://openreview.net/forum?id=rbHOLX8OWh
    - Role: common-feature-space motivation for cross-model latent transfer.
    - Boundary: SAE universality would support the next learned branch, not the
      current fixed-hybrid packet gate.

11. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
    - Link: https://arxiv.org/abs/2504.19874
    - Role: vector/KV quantization systems comparator.
    - Boundary: TurboQuant compresses dense vectors and KV state; this gate
      sends no continuous source vector.

12. TurboQuant: Redefining AI efficiency with extreme compression
    - Link: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
    - Role: current systems framing for memory-bottleneck quantization claims.
    - Boundary: LatentWire should compare against quantized-state byte floors,
      but should not claim to be a quantizer.

## Decision Boundary

Cite this artifact as:

```text
non-overlapping 1024-row physical candidate-text hardening for a
source-private fixed-hybrid HellaSwag candidate packet.
```

Do not cite it as:

```text
all-24-per-example permutation invariance
full-validation physical permutation invariance
learned cross-model latent language
native systems speedup
```
