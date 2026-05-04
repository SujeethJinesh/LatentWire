# HellaSwag Fixed-Hybrid True Candidate-Text Permutation References

Date: 2026-05-03

## Why This Memo Exists

This memo records the related-work boundary for the HellaSwag full hidden
fixed-hybrid true candidate-text permutation gate. Unlike the cached
option-position audit, this gate physically reorders candidate endings and
reruns the hidden-innovation source pipeline before remapping predictions back
to canonical candidate IDs.

## Primary Sources And Boundaries

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - Link: https://arxiv.org/abs/1905.07830
   - Role: benchmark for the fixed-hybrid packet and candidate-text
     permutation hardening.
   - Boundary: HellaSwag remains a multiple-choice continuation benchmark, so
     candidate-order controls are necessary before claiming robust reasoning
     transfer.

2. Large Language Models Are Not Robust Multiple Choice Selectors
   - Link: https://arxiv.org/abs/2309.03882
   - Role: primary warning that option-ID priors can distort MCQ evaluation.
   - Boundary: this gate directly tests candidate-text permutation for the
     hidden fixed-hybrid packet, but does not implement PriDe or eliminate the
     need for wider order controls.

3. Large Language Models Sensitivity to The Order of Options in
   Multiple-Choice Questions
   - Link: https://arxiv.org/abs/2308.11483
   - Role: documents that MCQ accuracy can change when answer options are
     reordered.
   - Boundary: this motivates physical candidate-text reruns rather than
     cached label remaps.

4. A Study on Large Language Models' Limitations in Multiple-Choice Question
   Answering
   - Link: https://arxiv.org/abs/2401.07955
   - Role: additional MCQ choice-order robustness warning.
   - Boundary: the current gate is a 512-row one-permutation-per-example
     hardening result, not all-permutation invariance.

5. Prefix-Tuning: Optimizing Continuous Prompts for Generation
   - Link: https://aclanthology.org/2021.acl-long.353/
   - Role: continuous prompt/prefix baseline.
   - Boundary: the fixed-hybrid packet emits a per-example source-conditioned
     candidate ID, not a persistent optimized soft prefix.

6. Learning to Compress Prompts with Gist Tokens
   - Link: https://arxiv.org/abs/2304.08467
   - Role: prompt-compression baseline.
   - Boundary: gist tokens compress visible prompt context; the current result
     sends no source text, source KV, raw hidden vector, score vector, or
     logits.

7. Cache-to-Cache: Direct Semantic Communication Between Large Language Models
   - Link: https://openreview.net/forum?id=LeatkxrBCi
   - Role: direct source KV/cache communication comparator.
   - Boundary: C2C projects and fuses source KV-cache state. LatentWire's
     current HellaSwag gate transmits a final candidate packet, so the novelty
     is extreme-rate source-private packet hardening, not KV semantic fusion.

8. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing
   - Link: https://openreview.net/forum?id=F7rUng23nw
   - Role: selective KV-sharing comparator.
   - Boundary: KVComm moves selected KV pairs; this gate moves no source KV
     state.

9. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
   Overhead
   - Link: https://arxiv.org/abs/2406.03482
   - Role: low-bit KV/vector compression comparator.
   - Boundary: QJL is relevant if LatentWire sends vectors or KV-like source
     state; this result sends a discrete candidate packet.

10. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
    - Link: https://arxiv.org/abs/2504.19874
    - Role: modern systems-side vector quantization comparator.
    - Boundary: TurboQuant pressures vector/KV transport claims, not a
      source-private final candidate packet.

11. Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment
    - Link: https://arxiv.org/abs/2502.03714
    - Role: possible next learned/common-basis method branch.
    - Boundary: the current permutation gate is not a sparse dictionary or
      crosscoder result; it only hardens the existing hidden packet.

12. Sparse Crosscoders for Cross-Layer Features and Model Diffing
    - Link: https://transformer-circuits.pub/2024/crosscoders/index.html
    - Role: inspiration for decision-supervised sparse hidden-innovation
      packets.
    - Boundary: a future LatentWire contribution must show task/decision
      supervision and source-destroying controls, not merely reconstruction
      overlap.

13. LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning
    - Link: https://arxiv.org/abs/2510.04573
    - Role: recent latent diffusion / iterative refinement motivation.
    - Boundary: the current HellaSwag gate is not diffusion reasoning. A
      future iterative score-simplex receiver would need to separate itself
      from diffusion-language modeling by operating on candidate belief states,
      not generated text.

## Decision Boundary

This gate should be cited as:

```text
512-row physical candidate-text hardening for the full hidden fixed-hybrid
HellaSwag candidate packet
```

It should not be cited as:

```text
all-permutation invariance
full-validation permutation invariance
learned receiver fusion
common latent-language evidence
native systems speedup
general cross-model reasoning transfer
```

The next reviewer-facing permutation gate is `1024` rows or all `24`
candidate permutations on `512` rows. The next method gate is a
decision-supervised SAE/crosscoder hidden-innovation packet with atom-shuffle,
wrong-row, and top-atom knockout controls.
