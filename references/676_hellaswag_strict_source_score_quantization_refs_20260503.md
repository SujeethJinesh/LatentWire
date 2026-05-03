# HellaSwag Strict Source-Score Quantization References

Date: 2026-05-03

## Why This Memo Exists

The strict source-score quantization gate tests whether score-vector, rank, and
margin codes can beat the `1B` candidate-only packet on the same HellaSwag
`0:9216` surface. They do not. This memo records the prior-work boundary for
using those rows as controls rather than positive-method claims.

## Primary Sources And Boundaries

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - Link: https://arxiv.org/abs/1905.07830
   - Role: multiple-choice validation surface for the strict score-code gate.
   - Boundary: the current result is format-specific and cannot by itself
     establish open-ended model-to-model reasoning.

2. Prefix-Tuning: Optimizing Continuous Prompts for Generation
   - Link: https://aclanthology.org/2021.acl-long.353/
   - Role: frozen-model continuous conditioning baseline.
   - Boundary: score-code packets are discrete, per-example, source-conditioned
     packets; they are not learned continuous prefixes.

3. Learning to Compress Prompts with Gist Tokens
   - Link: https://arxiv.org/abs/2304.08467
   - Role: prompt-compression baseline.
   - Boundary: gist tokens compress prompt context; the source-score gate tests
     whether compressed source model score evidence can improve target choice.

4. Relative Representations Enable Zero-Shot Latent Space Communication
   - Link: https://arxiv.org/abs/2209.15430
   - Role: common-coordinate latent communication prior.
   - Boundary: score-code quantization is not a new relative-representation
     method. It is a matched-byte control for source-score evidence.

5. Sparse Crosscoders for Cross-Layer Features and Model Diffing
   - Link: https://transformer-circuits.pub/2024/crosscoders/index.html
   - Role: shared sparse-feature/common-basis prior.
   - Boundary: the score-code gate has no SAE/crosscoder dictionary. If we use
     such dictionaries next, the novelty must be causal communication under
     source-destroying controls, not the existence of a shared basis.

6. Cache-to-Cache: Direct Semantic Communication Between Large Language Models
   - Link: https://arxiv.org/abs/2510.03215
   - Role: direct KV-cache fusion baseline.
   - Boundary: C2C sends/fuses source KV state. The score-code gate sends only
     compact score-derived symbols and is much lower exposure, but also much
     less expressive.

7. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing
   - Link: https://arxiv.org/abs/2510.03346
   - Role: selective KV-sharing systems baseline.
   - Boundary: KVComm is a state-transfer method. Source-score packets should
     be compared through byte/exposure/accuracy tables, not claimed as a cache
     replacement without native serving data.

8. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
   Overhead
   - Link: https://arxiv.org/abs/2406.03482
   - Role: low-bit vector/KV compression baseline.
   - Boundary: QJL preserves approximate attention geometry; our score-code
     gate tests task-score symbols. QJL is a compression floor for future
     source-state rows.

9. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
   - Link: https://arxiv.org/abs/2504.19874
   - Role: modern online vector quantization baseline.
   - Boundary: TurboQuant pressures any byte-efficiency claim involving
     transmitted vectors/KV. It does not preempt a source-private task-packet
     result, but it must appear in the systems comparison table.

## Decision Boundary

The score-code gate should be cited as a negative control:

```text
calibrated source-score code <= source argmax << candidate-only hidden packet
```

This means the current HellaSwag packet is not explained by raw score-vector
quantization alone. It also means score-only branches are lower expected value
than target-loss query/soft-prefix connectors or conditional hidden-innovation
packets for the next ICLR method gate.
