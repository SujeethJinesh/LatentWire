# References: HellaSwag Official-Train Receiver Calibration

## Claim Boundary

This memo supports the official-train receiver-calibration artifact. The safe
claim is that validation-label-free scalar acceptance still fails to capture
the TinyLlama/Qwen complementary-error headroom on full HellaSwag. It does not
claim solved cross-model latent reasoning or native systems superiority.

## Primary Sources And Why They Matter

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - https://arxiv.org/abs/1905.07830
   - Why it matters: the frozen full-validation benchmark surface.

2. TinyLlama: An Open-Source Small Language Model
   - https://arxiv.org/abs/2401.02385
   - Why it matters: the non-Qwen source family for the packet-only row.

3. Selective Classification for Deep Neural Networks and SelectiveNet
   - https://arxiv.org/abs/1705.08500
   - https://arxiv.org/abs/1901.09192
   - Boundary: the artifact tests a selective override problem. The failure
     rules out cheap scalar selectors, not all possible learned receivers.

4. Relative Representations Enable Zero-Shot Latent Space Communication
   - https://openreview.net/forum?id=SrC-nwieGJ
   - Why it matters: anchor-relative coordinates motivate common-basis
     receivers, but this artifact shows relative-kNN acceptance is not enough.

5. Sparse Autoencoders and Universal Feature Spaces
   - https://arxiv.org/abs/2309.08600
   - https://arxiv.org/abs/2410.06981
   - Why it matters: SAE/dictionary methods are a plausible next branch for
     reusable disagreement atoms after scalar acceptance fails.

6. BLIP-2 Q-Former, Flamingo, and Perceiver IO
   - https://arxiv.org/abs/2301.12597
   - https://arxiv.org/abs/2204.14198
   - https://arxiv.org/abs/2107.14795
   - Why it matters: learned query bottlenecks are the strongest architectural
     precedent for a frozen-model bridge that does not require a shared native
     basis.

7. Prefix-Tuning, P-Tuning v2, and LLaMA-Adapter
   - https://aclanthology.org/2021.acl-long.353/
   - https://arxiv.org/abs/2110.07602
   - https://arxiv.org/abs/2303.16199
   - Boundary: prefix/prompt/adapter methods learn persistent conditioning
     vectors or adapters. LatentWire's current packet is per-example,
     source-private, fixed-byte, and transmitted without target weight updates.

8. Cache-to-Cache and KV Communication
   - https://arxiv.org/abs/2510.03215
   - https://arxiv.org/abs/2510.12872
   - https://arxiv.org/abs/2512.17914
   - Boundary: C2C/KVComm/Q-KVComm communicate through projected, fused, reused,
     or compressed KV/cache state. This artifact sends no source KV, raw hidden
     vector, raw source score vector, or source text.

9. QJL, KIVI, KVQuant, and TurboQuant
   - https://arxiv.org/abs/2406.03482
   - https://arxiv.org/abs/2402.02750
   - https://arxiv.org/abs/2401.18079
   - https://arxiv.org/abs/2504.19874
   - Boundary: these are vector/KV compression methods. They motivate byte-floor
     and rate-distortion comparisons, but the current artifact is a task-level
     source-private packet receiver, not KV compression.

10. Diffusion Transformers
    - https://arxiv.org/abs/2212.09748
    - Why it matters: DiT-style denoising suggests an iterative repair analogy,
      but this artifact tests one-shot scalar acceptance. A diffusion/flow
      bridge would be a new method branch, not an explanation of this result.

11. vLLM / PagedAttention and SGLang
    - https://arxiv.org/abs/2309.06180
    - https://arxiv.org/abs/2312.07104
    - Boundary: native serving comparisons remain pending until NVIDIA GPU
      rows are available.

## Reviewer-Facing Framing

Safe:

- The artifact removes validation-label calibration from the receiver branch.
- Official-train scalar acceptance still fails to beat packet-only, despite a
  large Tiny/Qwen oracle headroom.
- The next receiver should change representation structure: disagreement
  prototypes, sparse/crosscoder dictionaries, or learned query bottlenecks.

Unsafe:

- Claiming receiver improvement over packet-only.
- Claiming this is equivalent to prefix/prompt tuning or to C2C/KVComm.
- Claiming systems superiority over QJL, KIVI, KVQuant, TurboQuant, vLLM, or
  SGLang before native runs.
