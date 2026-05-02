# References: HellaSwag Disagreement-Prototype Receiver

## Claim Boundary

This memo supports the train-only disagreement-prototype receiver artifact.
The safe claim is that local disagreement prototypes are a tested common-basis
receiver branch that did not clear the receiver-improvement gate. It does not
claim solved cross-model latent reasoning, first latent communication, or
native systems superiority.

## Primary Sources And Why They Matter

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - https://arxiv.org/abs/1905.07830
   - Why it matters: the frozen full-validation benchmark surface.

2. TinyLlama: An Open-Source Small Language Model
   - https://arxiv.org/abs/2401.02385
   - Why it matters: the non-Qwen source-family packet-only row.

3. Selective Classification and SelectiveNet
   - https://arxiv.org/abs/1705.08500
   - https://arxiv.org/abs/1901.09192
   - Boundary: this artifact is an accept/override receiver, but the method
     sends a fixed-byte source-private packet and calibrates cross-model
     disagreement. It is not a generic reject-option classifier.

4. Relative Representations Enable Zero-Shot Latent Space Communication
   - https://arxiv.org/abs/2209.15430
   - Boundary: relative representations use anchor-relative coordinates to
     stabilize latent comparisons. This artifact tests a much smaller
     receiver-side prototype estimate over task features and fixed-byte
     packets; it does not transmit or align dense latent vectors.

5. Sparse Autoencoders, Universal Feature Spaces, and Sparse Crosscoders
   - https://arxiv.org/abs/2309.08600
   - https://arxiv.org/abs/2410.06981
   - https://transformer-circuits.pub/2024/crosscoders/index.html
   - Why it matters: these are the natural next branch after prototype failure,
     because they can represent reusable disagreement atoms rather than scalar
     or nearest-prototype thresholds.

6. Prefix-Tuning, P-Tuning v2, and LLaMA-Adapter
   - https://aclanthology.org/2021.acl-long.353/
   - https://arxiv.org/abs/2110.07602
   - https://arxiv.org/abs/2303.16199
   - Boundary: prefix/prompt/adapters learn persistent conditioning vectors or
     modules. LatentWire's packet is per-example, fixed-byte, source-private,
     and does not inject virtual tokens or update the receiver model.

7. Cache-to-Cache and KV Communication
   - https://openreview.net/forum?id=LeatkxrBCi
   - https://arxiv.org/abs/2510.03346
   - https://arxiv.org/abs/2510.12872
   - https://arxiv.org/abs/2512.17914
   - Boundary: C2C/KVComm/KVCOMM/Q-KVComm communicate by projecting, sharing,
     reusing, or compressing KV/cache state. This artifact sends no source KV,
     raw hidden vector, raw score vector, or source text.

8. QJL, KIVI, KVQuant, and TurboQuant
   - https://arxiv.org/abs/2406.03482
   - https://arxiv.org/abs/2402.02750
   - https://arxiv.org/abs/2401.18079
   - https://arxiv.org/abs/2504.19874
   - Boundary: these are KV/vector compression methods. They motivate
     rate-distortion and byte-floor comparisons, but the current artifact
     transmits a semantic decision packet rather than compressed KV state.

9. BLIP-2 Q-Former, Flamingo, and Perceiver IO
   - https://arxiv.org/abs/2301.12597
   - https://arxiv.org/abs/2204.14198
   - https://arxiv.org/abs/2107.14795
   - Why it matters: learned query bottlenecks are a plausible next receiver
     architecture after scalar and prototype receivers fail.

10. Diffusion Transformers and Latent Diffusion
    - https://arxiv.org/abs/2212.09748
    - https://arxiv.org/abs/2112.10752
    - Boundary: diffusion-style denoising is a possible future repair branch.
      The current artifact is one-shot receiver selection, not iterative
      latent generation or denoising.

11. vLLM / PagedAttention and SGLang
    - https://arxiv.org/abs/2309.06180
    - https://arxiv.org/abs/2312.07104
    - Boundary: native serving comparisons remain pending until NVIDIA GPU
      rows are available.

## Reviewer-Facing Framing

Safe:

- The artifact tests whether a train-only local common-basis/prototype
  receiver can close TinyLlama/Qwen complementary-error headroom.
- The predeclared receiver fails, and the best scout is too small
  (`+0.001494`) to promote.
- The failure narrows the next receiver branch toward sparse/crosscoder
  dictionaries or learned query bottlenecks.

Unsafe:

- Claiming receiver improvement over packet-only.
- Claiming novelty over all latent communication or representation alignment.
- Claiming equivalence to prefix tokens, C2C, KVComm, or KV quantization.
- Claiming systems superiority before native NVIDIA/vLLM/SGLang rows.
