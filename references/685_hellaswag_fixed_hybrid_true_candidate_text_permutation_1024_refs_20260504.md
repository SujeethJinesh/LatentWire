# HellaSwag Fixed-Hybrid True Candidate-Text Permutation 1024 References

Date: 2026-05-04

## Why This Memo Exists

This memo records the related-work boundary for the `1024`-row HellaSwag
physical candidate-text permutation rerun of the full hidden fixed-hybrid
packet. The result strengthens the option-order defense for the current
source-private packet row, but it is not a learned common-basis or native
systems contribution.

## Primary Sources And Boundaries

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - Link: https://arxiv.org/abs/1905.07830
   - Role: benchmark for adversarially filtered commonsense continuation.
   - Boundary: HellaSwag is multiple choice, so candidate-order hardening is
     necessary before claiming a robust source packet.

2. Large Language Models Are Not Robust Multiple Choice Selectors
   - Link: https://arxiv.org/abs/2309.03882
   - Role: option-ID and selector-bias warning.
   - Boundary: this gate physically reruns shuffled candidate text, but still
     does not eliminate all MCQ-order sensitivity.

3. Large Language Models Sensitivity to The Order of Options in
   Multiple-Choice Questions
   - Link: https://arxiv.org/abs/2308.11483
   - Role: motivates physical candidate reordering rather than cached remaps.
   - Boundary: the current result uses one non-identity permutation per row,
     not all `24` permutations.

4. Prefix-Tuning: Optimizing Continuous Prompts for Generation
   - Link: https://aclanthology.org/2021.acl-long.353/
   - Role: continuous prompt comparator.
   - Boundary: prefix tuning learns persistent task conditioning; this gate
     evaluates per-example source-conditioned candidate packets.

5. Learning to Compress Prompts with Gist Tokens
   - Link: https://arxiv.org/abs/2304.08467
   - Role: prompt/context compression comparator.
   - Boundary: gist tokens compress visible context; the packet transmits no
     source text, source KV, raw hidden vector, score vector, or logits.

6. Cache-to-Cache: Direct Semantic Communication Between Large Language Models
   - Link: https://arxiv.org/abs/2510.03215
   - Role: closest direct model-to-model cache communication comparator.
   - Boundary: C2C transmits/fuses high-dimensional source KV-derived state.
     The current result transmits only a final fixed-hybrid candidate packet.

7. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing
   - Link: https://arxiv.org/abs/2510.03346
   - Role: selective source-KV communication comparator.
   - Boundary: KVComm moves selected KV state; this gate moves no source KV.

8. Online KVCOMM Cross-Context KV-Cache Communication
   - Link: https://arxiv.org/abs/2510.12872
   - Role: cross-context KV reuse and serving-efficiency comparator.
   - Boundary: KVCOMM is a cache-reuse systems method, not a source-private
     task-evidence packet protocol.

9. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
   - Link: https://arxiv.org/abs/2504.19874
   - Role: vector/KV quantization pressure test for any latent-state transport
     claim.
   - Boundary: TurboQuant compresses continuous state; this gate transmits a
     discrete decision packet.

10. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
    Overhead
    - Link: https://arxiv.org/abs/2406.03482
    - Role: aggressive KV/state byte-floor comparator.
    - Boundary: QJL still represents source/target KV-like state, while the
      current packet exposes no source hidden/KV vector.

11. Sparse Crosscoders for Cross-Layer Features and Model Diffing
    - Link: https://transformer-circuits.pub/2024/crosscoders/index.html
    - Role: inspiration for the next common-basis packet branch.
    - Boundary: crosscoders are not yet the current claim. A future branch
      must beat the fixed packet under atom-shuffle, wrong-row, and top-atom
      knockout controls.

12. Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment
    - Link: https://arxiv.org/abs/2502.03714
    - Role: common sparse feature-space motivation.
    - Boundary: SAE alignment is a plausible next method; this artifact only
      hardens the existing packet-policy row.

## Decision Boundary

Cite this artifact as:

```text
1024-row physical candidate-text hardening for a source-private HellaSwag
fixed-hybrid candidate packet.
```

Do not cite it as:

```text
all-permutation invariance
full-validation permutation invariance
learned cross-model latent language
native hardware speedup
general reasoning transfer
```
