# HellaSwag Score-Channel True Candidate-Text Permutation References

Date: 2026-05-03

## Why This Memo Exists

This memo records the related-work boundary for the HellaSwag
score-channel true candidate-text permutation gate. The new artifact physically
reruns the Qwen continuation-likelihood source scorer after candidate endings
are reordered, then remaps predictions back to canonical candidate IDs.

This is candidate-text hardening for one component of the current packet stack.
It is not a full fixed-hybrid hidden-innovation rerun and not evidence for a
learned common latent language.

## Primary Sources And Boundaries

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - Link: https://arxiv.org/abs/1905.07830
   - Role: benchmark used for the candidate-text permutation smoke.
   - Boundary: HellaSwag is a multiple-choice continuation benchmark; option
     ordering remains an evaluator threat unless tested directly.

2. Large Language Models Are Not Robust Multiple Choice Selectors
   - Link: https://arxiv.org/abs/2309.03882
   - Role: primary warning that option-ID priors can distort MCQ evaluation and
     that option-content permutation can be used to estimate/debias priors.
   - Boundary: the new gate is not PriDe; it is a direct physical permutation
     check for the score-channel source packet.

3. Large Language Models Sensitivity to The Order of Options in
   Multiple-Choice Questions
   - Link: https://arxiv.org/abs/2308.11483
   - Role: primary warning that answer option order can materially change MCQ
     accuracy.
   - Boundary: this motivates true candidate-text reruns rather than cached
     label remaps.

4. A Study on Large Language Models' Limitations in Multiple-Choice Question
   Answering
   - Link: https://arxiv.org/abs/2401.07955
   - Role: additional MCQ choice-order robustness warning.
   - Boundary: supports strict scope: a 128-row component smoke is hardening,
     not final benchmark invariance.

5. When Benchmarks are Targets: Revealing the Sensitivity of Large Language
   Model Leaderboards
   - Link: https://arxiv.org/abs/2402.01781
   - Role: shows benchmark rankings can shift under small evaluation
     perturbations, including option-order/scoring choices.
   - Boundary: reinforces that HellaSwag packet claims need transparent
     evaluator controls.

6. Prefix-Tuning: Optimizing Continuous Prompts for Generation
   - Link: https://aclanthology.org/2021.acl-long.353/
   - Role: continuous-prefix baseline.
   - Boundary: LatentWire's current HellaSwag packet sends a discrete
     source-private candidate ID, not optimized task-specific prefix vectors.

7. Learning to Compress Prompts with Gist Tokens
   - Link: https://arxiv.org/abs/2304.08467
   - Role: prompt-compression baseline.
   - Boundary: gist tokens compress visible prompt context; this gate checks a
     source-private score-channel packet with no source text, KV, hidden vector,
     score vector, or logits transmitted.

8. Cache-to-Cache: Direct Semantic Communication Between Large Language Models
   - Link: https://openreview.net/forum?id=LeatkxrBCi
   - Role: direct source KV/cache fusion comparator.
   - Boundary: C2C transfers and fuses source KV-cache state. The current gate
     transfers a one-byte candidate ID and therefore should not claim C2C-like
     semantic cache transfer.

9. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing
   - Link: https://openreview.net/forum?id=F7rUng23nw
   - Role: selective KV-sharing systems comparator.
   - Boundary: KVComm moves selected KV pairs; this gate moves no source KV or
     hidden state.

10. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
    Overhead
    - Link: https://arxiv.org/abs/2406.03482
    - Role: low-bit KV/vector compression comparator.
    - Boundary: QJL pressures any future vector/cache transport claim; it is
      not the same object as a source-private candidate packet.

11. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
    - Link: https://arxiv.org/abs/2504.19874
    - Role: modern near-optimal online vector quantization comparator for
      systems-side compression claims.
    - Boundary: TurboQuant is relevant when LatentWire sends vectors or KV-like
      source state. The current result sends no raw vector state.

## Decision Boundary

This gate should be cited as:

```text
true candidate-text permutation hardening for the score-channel candidate-id
packet component
```

It should not be cited as:

```text
full fixed-hybrid hidden permutation invariance
learned receiver fusion
common latent-language evidence
native systems speedup
general cross-model reasoning transfer
```

The next stronger HellaSwag control is a hidden fixed-hybrid physical
candidate-text permutation rerun over at least `512` frozen rows, with canonical
remapping, paired uncertainty, and wrong-remap controls.
