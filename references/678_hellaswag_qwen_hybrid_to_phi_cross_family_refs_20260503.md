# HellaSwag Qwen Hybrid-To-Phi Cross-Family References

Date: 2026-05-03

## Why This Memo Exists

The cached Qwen-to-Phi cross-family gate shows that the fixed Qwen hybrid
vote-on-score-agreement packet policy improves over Qwen candidate-only on
Phi-heldout HellaSwag rows. This memo records the novelty boundary: this is
packet-policy survival under cross-family pressure, not a learned latent
receiver or common-basis method.

## Primary Sources And Boundaries

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - Link: https://arxiv.org/abs/1905.07830
   - Role: multiple-choice commonsense benchmark used by the cross-family gate.
   - Boundary: the result remains answer-choice scoped and should not be
     generalized to open-ended reasoning without additional benchmarks.

2. Large Language Models Are Not Robust Multiple Choice Selectors
   - Link: https://arxiv.org/abs/2309.03882
   - Role: option-id and candidate-selection robustness warning.
   - Boundary: this motivates candidate-roll, label, source-rank, and
     same-byte controls. It also means the paper must avoid treating
     multiple-choice candidate transfer as general reasoning transfer.

3. Relative Representations Enable Zero-Shot Latent Space Communication
   - Link: https://openreview.net/forum?id=SrC-nwieGJ
   - Role: common-coordinate latent communication prior.
   - Boundary: the Qwen-to-Phi hybrid packet does not construct a relative
     representation. It sends one source-chosen candidate id.

4. Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language
   Models
   - Link: https://arxiv.org/abs/2410.06981
   - Role: evidence that some LLM feature spaces may have shared structure.
   - Boundary: SAE universality is future motivation only. The current gate
     does not expose or align SAE features.

5. Crosscoder Model Diffing
   - Link: https://www.anthropic.com/research/crosscoder-model-diffing
   - Role: shared-feature/model-diffing prior for future common-basis methods.
   - Boundary: the current packet-policy result is not a crosscoder or
     feature-dictionary method.

6. Prefix-Tuning: Optimizing Continuous Prompts for Generation
   - Link: https://aclanthology.org/2021.acl-long.353/
   - Role: frozen-model continuous conditioning baseline.
   - Boundary: Phi receives no learned soft prefix in this gate.

7. Learning to Compress Prompts with Gist Tokens
   - Link: https://arxiv.org/abs/2304.08467
   - Role: prompt-compression baseline.
   - Boundary: gist tokens compress prompt context; the Qwen-to-Phi gate sends
     source-side task evidence as a final candidate id.

8. Cache-to-Cache: Direct Semantic Communication Between Large Language Models
   - Link: https://openreview.net/forum?id=LeatkxrBCi
   - Role: direct KV-cache fusion baseline.
   - Boundary: C2C is much more expressive and exposes/fuses source state. The
     hybrid packet is lower exposure but should not be claimed as a general
     replacement without native systems rows and broader tasks.

9. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing
   - Link: https://openreview.net/forum?id=F7rUng23nw
   - Role: selective KV-sharing comparator for inter-LLM communication.
   - Boundary: KVComm transfers source KV pairs; this gate transfers only a
     candidate id.

10. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
    Overhead
    - Link: https://arxiv.org/abs/2406.03482
    - Role: low-bit KV/vector-state compression floor.
    - Boundary: QJL pressures future vector/KV-state rows, not the discrete
      packet row directly.

11. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
    - Link: https://arxiv.org/abs/2504.19874
    - Role: modern online vector quantization baseline.
    - Boundary: TurboQuant is a systems compression comparator if LatentWire
      transmits vector state. It does not preempt a one-candidate task packet.

## Decision Boundary

This gate should be cited as:

```text
cross-family survival of a source-private fixed-byte packet policy
```

It should not be cited as:

```text
learned Phi receiver fusion
common latent basis
SAE/crosscoder communication
prefix/gist-token compression
native systems speedup
```

The next ICLR-grade step is to beat this row with a packet-preserving
anti-harm veto, target-loss soft-prefix connector, or conditional
hidden-innovation receiver under the same destructive controls.
