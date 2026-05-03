# HellaSwag Fixed Hybrid Option-Position Audit References

Date: 2026-05-03

## Why This Memo Exists

The fixed hybrid packet now has a cached option-position and packet-ID
permutation audit over full HellaSwag validation `0:10042`. This memo records
the related-work boundary: the audit hardens a candidate-id packet result
against simple slot-prior explanations, but it is not a substitute for true
candidate-text permutation reruns.

## Primary Sources And Boundaries

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - Link: https://arxiv.org/abs/1905.07830
   - Role: benchmark used for the full-validation fixed hybrid packet and
     option-position audit.
   - Boundary: HellaSwag remains a multiple-choice commonsense benchmark; a
     candidate-id packet result needs option-order controls before broader
     reasoning-transfer claims.

2. Large Language Models Are Not Robust Multiple Choice Selectors
   - Link: https://arxiv.org/abs/2309.03882
   - Role: establishes option-ID prior bias as a direct threat to MCQ
     evaluation.
   - Boundary: this audit checks answer-position stratification, cyclic
     packet-roll controls, non-identity packet-label remaps, and rowwise
     derangements, but does not implement PriDe or eliminate the need for
     candidate-text permutation reruns.

3. Large Language Models Sensitivity to The Order of Options in
   Multiple-Choice Questions
   - Link: https://arxiv.org/abs/2308.11483
   - Role: documents that MCQ accuracy can change substantially when answer
     options are reordered.
   - Boundary: the current cached audit cannot measure source-model behavior
     under reordered candidate text. A true permutation rerun remains the
     stronger future gate.

4. A Study on Large Language Models' Limitations in Multiple-Choice Question
   Answering
   - Link: https://arxiv.org/abs/2401.07955
   - Role: additional MCQ robustness warning, including choice-order
     independence.
   - Boundary: candidate-id packet claims should be scoped until choice-order
     independence is tested directly.

5. Cache-to-Cache: Direct Semantic Communication Between Large Language Models
   - Link: https://openreview.net/forum?id=LeatkxrBCi
   - Role: high-rate source-state communication baseline.
   - Boundary: C2C transfers source KV/cache state, while this audit checks a
     one-candidate fixed-byte packet. The audit does not close systems or
     expressivity gaps against C2C.

6. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing
   - Link: https://openreview.net/forum?id=F7rUng23nw
   - Role: selective KV-sharing communication comparator.
   - Boundary: KVComm sends selected KV state. The fixed hybrid packet sends no
     KV state and needs native systems rows before systems superiority claims.

7. Prefix-Tuning: Optimizing Continuous Prompts for Generation
   - Link: https://aclanthology.org/2021.acl-long.353/
   - Role: continuous-prefix baseline.
   - Boundary: this audit does not optimize soft prompts or latent prefixes.

8. Learning to Compress Prompts with Gist Tokens
   - Link: https://arxiv.org/abs/2304.08467
   - Role: prompt-compression baseline.
   - Boundary: gist tokens compress visible prompt context; LatentWire's
     current HellaSwag result transmits a source-private candidate packet.

9. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
   Overhead
   - Link: https://arxiv.org/abs/2406.03482
   - Role: low-bit KV/vector compression floor.
   - Boundary: QJL is a comparator for future source-state/vector transport,
     not for a one-candidate packet-position audit.

10. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
    - Link: https://arxiv.org/abs/2504.19874
    - Role: quantization and systems-compression comparator.
    - Boundary: TurboQuant pressures latent/vector communication claims and
      native systems comparisons; it does not explain the cached
      option-position audit.

## Decision Boundary

This audit should be cited as:

```text
cached option-position and packet-ID permutation hardening for a
full-validation source-private candidate packet row
```

It should not be cited as:

```text
true candidate-text permutation invariance
learned receiver fusion
common latent basis
native systems speedup
general reasoning transfer
```

The next HellaSwag bias-control gate should physically reorder candidate texts
on a frozen slice and remap predictions back to canonical IDs.
