# HellaSwag Fixed-Hybrid Seeded Candidate-Text Permutation References

Date: 2026-05-04

## Why This Memo Exists

This memo records the related-work and novelty boundary for the `1024`-row
seed-controlled HellaSwag physical candidate-text permutation gate. The gate
fixes the earlier no-op `--seed` path and verifies the fixed-hybrid packet
under a distinct rowwise permutation schedule.

## Primary Sources And Boundaries

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - Link: https://arxiv.org/abs/1905.07830
   - Role: benchmark for adversarially filtered commonsense continuation.
   - Boundary: HellaSwag remains a multiple-choice benchmark, so candidate
     option-order hardening is required before making robust communication
     claims.

2. Large Language Models Are Not Robust Multiple Choice Selectors
   - Link: https://arxiv.org/abs/2309.03882
   - Role: option-ID/selector-bias warning.
   - Boundary: the current seeded physical shuffle is a direct hardening
     control, but still not a complete solution to all MCQ-order effects.

3. Large Language Models Sensitivity to The Order of Options in
   Multiple-Choice Questions
   - Link: https://arxiv.org/abs/2308.11483
   - Role: motivates physical candidate-text reordering rather than cached
     label remaps.
   - Boundary: this gate uses one seeded non-identity permutation per row, not
     all `24` candidate orders per row.

4. Cache-to-Cache: Direct Semantic Communication Between Large Language Models
   - Link: https://arxiv.org/abs/2510.03215
   - Role: closest direct semantic model-to-model communication comparator.
   - Boundary: C2C projects and fuses source KV-cache state; this gate
     transmits only a final source-private candidate packet.

5. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing
   - Link: https://arxiv.org/abs/2510.03346
   - Role: selective KV-sharing comparator.
   - Boundary: KVComm moves selected KV pairs/layers; this gate moves no
     source KV, hidden vector, score vector, or logits.

6. Enabling Agents to Communicate Entirely in Latent Space
   - Link: https://arxiv.org/abs/2511.09149
   - Role: latent hidden-state communication comparator.
   - Boundary: Interlat transmits generated-token hidden states; LatentWire's
     current gate sends a byte-scale decision packet under destructive
     controls.

7. Prefix-Tuning: Optimizing Continuous Prompts for Generation
   - Link: https://aclanthology.org/2021.acl-long.353/
   - Role: soft-prefix baseline.
   - Boundary: prefix tuning learns persistent task conditioning rather than a
     per-example source-private packet.

8. Learning to Compress Prompts with Gist Tokens
   - Link: https://arxiv.org/abs/2304.08467
   - Role: prompt/context compression baseline.
   - Boundary: gist tokens compress visible context into reusable prompt
     tokens; the current packet does not expose source text or source state.

9. Relative representations enable zero-shot latent space communication
   - Link: https://openreview.net/forum?id=SrC-nwieGJ
   - Role: common-coordinate latent-space comparator.
   - Boundary: relative representations align latent spaces through anchors;
     this gate is not a common-basis method.

10. Sparse Crosscoders for Cross-Layer Features and Model Diffing
    - Link: https://transformer-circuits.pub/2024/crosscoders/index.html
    - Role: inspiration for the next learned common-basis packet branch.
    - Boundary: crosscoders identify shared/exclusive sparse features; the
      current result only hardens a fixed packet-policy row.

11. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
    - Link: https://openreview.net/forum?id=tO3ASKZlok
    - Role: vector/KV quantization systems comparator.
    - Boundary: TurboQuant compresses continuous vectors/KV state; this gate
      transmits no continuous source vector.

12. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
    Overhead
    - Link: https://arxiv.org/abs/2406.03482
    - Role: aggressive KV-cache byte-floor comparator.
    - Boundary: QJL is relevant for KV/state compression, not final
      source-private decision packets.

## Decision Boundary

Cite this artifact as:

```text
seed-controlled 1024-row physical candidate-text hardening for a
source-private fixed-hybrid HellaSwag candidate packet.
```

Do not cite it as:

```text
all-24-per-example permutation invariance
full-validation permutation invariance
learned cross-model latent language
native systems speedup
```
