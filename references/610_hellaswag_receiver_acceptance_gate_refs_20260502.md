# References: HellaSwag Receiver Acceptance Gate

## Claim Boundary

This memo supports a receiver-acceptance branch-kill artifact. The safe claim
is that train-only selective receiver baselines fail to capture the
TinyLlama/Qwen oracle headroom on HellaSwag. It does not claim solved
cross-model latent reasoning.

## Primary Sources And Why They Matter

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - https://arxiv.org/abs/1905.07830
   - Why it matters: the frozen full-validation benchmark surface.

2. TinyLlama: An Open-Source Small Language Model
   - https://arxiv.org/abs/2401.02385
   - Why it matters: the non-Qwen source family for the packet-only row.

3. Selective Classification for Deep Neural Networks
   - https://arxiv.org/abs/1705.08500
   - Why it matters: the receiver acceptance question is a selective decision:
     choose when to override the packet. Our train-only selective baselines do
     not clear packet-only.

4. SelectiveNet: A Deep Neural Network with an Integrated Reject Option
   - https://arxiv.org/abs/1901.09192
   - Why it matters: a stronger selective-prediction family exists, so the
     current ridge/kNN failure should be framed as ruling out cheap selectors,
     not all possible learned receivers.

5. Relative Representations Enable Zero-Shot Latent Space Communication
   - https://openreview.net/forum?id=SrC-nwieGJ
   - Why it matters: nearest-anchor relative coordinates are the mathematical
     common-basis inspiration behind the relative-kNN receiver row.

6. Sparse Autoencoders Find Highly Interpretable Features in Language Models
   - https://arxiv.org/abs/2309.08600
   - Why it matters: sparse dictionaries remain a plausible next branch after
     simple receiver acceptance fails.

7. Crosscoders
   - https://transformer-circuits.pub/2025/crosscoder-diffing/index.html
   - Why it matters: crosscoders target shared versus model-specific features,
     which is the precise failure mode exposed by complementary Qwen/TinyLlama
     errors.

8. Prefix-Tuning and Prompt Tuning
   - https://aclanthology.org/2021.acl-long.353/
   - https://arxiv.org/abs/2104.08691
   - Boundary: prompt/prefix methods learn persistent conditioning vectors;
     this artifact evaluates per-example source-private packets and a
     target-side acceptance rule.

9. Cache-to-Cache and KVComm
   - https://arxiv.org/abs/2510.03215
   - https://arxiv.org/abs/2510.03346
   - Boundary: these communicate through projected/fused or selectively shared
     KV/cache state. This artifact sends no source KV, raw hidden vector, raw
     source score vector, or source text.

10. QJL, KIVI, KVQuant, and TurboQuant
    - https://arxiv.org/abs/2406.03482
    - https://arxiv.org/abs/2402.02750
    - https://arxiv.org/abs/2401.18079
    - https://arxiv.org/abs/2504.19874
    - Boundary: these are vector/KV compression methods. They motivate systems
      byte-floor comparisons, but the current receiver gate is a task-level
      sideband packet rather than KV compression.

11. BLIP-2 Q-Former and Perceiver IO
    - https://arxiv.org/abs/2301.12597
    - https://arxiv.org/abs/2107.14795
    - Why it matters: learned query bottlenecks are a natural next receiver
      architecture after ridge and nearest-anchor selectors fail.

12. vLLM and SGLang
    - https://arxiv.org/abs/2309.06180
    - https://arxiv.org/abs/2312.07104
    - Boundary: native serving comparisons remain pending until NVIDIA GPU
      rows are available.

## Reviewer-Facing Framing

Safe:

- The artifact tests a direct selective receiver hypothesis under train/dev
  and heldout-suffix separation.
- The simple receiver families are weakened or ruled out on this surface.
- The result motivates official-train calibration or learned common-basis
  receivers, not more threshold tuning.

Unsafe:

- Claiming cross-model latent reasoning is solved.
- Claiming the receiver beats packet-only.
- Claiming native systems superiority over C2C, KVComm, KIVI, KVQuant, QJL,
  TurboQuant, vLLM, or SGLang.
