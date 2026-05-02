# References: HellaSwag Receiver-Family Packet Gate

## Claim Boundary

The receiver-family scout supports a target-family utility claim: a TinyLlama
source-private hidden packet is useful to a Qwen target-score receiver relative
to target-only scoring. It does not yet prove that the receiver learned a
cross-family latent language, because the selected receiver does not beat
packet-only.

## Primary Sources And Why They Matter

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - https://arxiv.org/abs/1905.07830
   - Why it matters: benchmark surface for the receiver-family scout.

2. TinyLlama: An Open-Source Small Language Model
   - https://arxiv.org/abs/2401.02385
   - Why it matters: non-Qwen source family that emits the hidden-innovation
     packet in this scout.

3. Prefix-Tuning: Optimizing Continuous Prompts for Generation
   - https://aclanthology.org/2021.acl-long.353/
   - Boundary: prefix tuning learns continuous vectors that condition a frozen
     model. LatentWire sends a per-example discrete packet derived from source
     hidden evidence.

4. The Power of Scale for Parameter-Efficient Prompt Tuning
   - https://arxiv.org/abs/2104.08691
   - Boundary: prompt tuning is a learned soft-prompt adaptation method, not a
     source-private packet sent between model families.

5. Cache-to-Cache: Direct Semantic Communication Between Large Language Models
   - https://arxiv.org/abs/2510.03215
   - Boundary: C2C communicates by fusing projected KV/cache state. This scout
     transmits no source KV, raw hidden vector, raw score vector, or source
     text.

6. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing
   - https://arxiv.org/abs/2510.03346
   - Boundary: KVComm is a closer inter-model systems baseline because it
     shares selected KV pairs. LatentWire is currently a stricter byte/exposure
     point, not a native KVComm throughput winner.

7. Communicating Activations Between Language Model Agents
   - https://arxiv.org/abs/2501.14082
   - Boundary: activation communication shares internal activations as the
     communication medium. LatentWire's current packet does not transmit raw
     activations, so the safe novelty claim is rate/exposure constrained task
     evidence rather than first activation-language communication.

8. Let Models Speak Ciphers: Multiagent Debate through Embeddings
   - https://arxiv.org/abs/2310.06272
   - Boundary: CIPHER studies embedding-based multi-agent debate. LatentWire is
     not the first non-natural-language model communication result; its
     distinct point is a fixed-byte source-private HellaSwag packet with
     destructive packet controls.

9. Relative Representations Enable Zero-Shot Latent Space Communication
   - https://openreview.net/pdf?id=SrC-nwieGJ
   - Why it matters: anchor-relative coordinates are a direct common-basis
     inspiration for the next receiver branch.

10. Sparse Autoencoders Find Highly Interpretable Features in Language Models
   - https://arxiv.org/abs/2309.08600
   - Why it matters: sparse dictionaries are a plausible route to a shared
     receiver-readable basis, but this scout does not train one.

11. Crosscoders
   - https://transformer-circuits.pub/2025/crosscoder-diffing/index.html
   - Why it matters: crosscoders explicitly separate shared and
     model-specific features, matching the current need for a common-language
     receiver branch.

12. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image
    Encoders and Large Language Models
    - https://arxiv.org/abs/2301.12597
    - Why it matters: the Q-Former pattern is a proven learned connector
      between frozen heterogeneous models; it motivates a target-aware
      receiver rather than a raw hidden-space map.

13. Perceiver IO
    - https://arxiv.org/abs/2107.14795
    - Why it matters: learned latent queries provide a bounded transport
      bottleneck for heterogeneous inputs and outputs.

14. QJL, KIVI, KVQuant, and TurboQuant
    - https://arxiv.org/abs/2406.03482
    - https://arxiv.org/abs/2402.02750
    - https://arxiv.org/abs/2401.18079
    - https://arxiv.org/abs/2504.19874
    - Boundary: these are KV/cache sketching or quantization systems
      comparators. They motivate the systems byte floor but do not duplicate a
      `2B` source-private task packet.

15. vLLM and SGLang
    - https://arxiv.org/abs/2309.06180
    - https://arxiv.org/abs/2312.07104
    - Boundary: native serving comparisons remain future work on NVIDIA GPUs.

## Reviewer-Facing Framing

Safe:

- The receiver-family scout shows that the TinyLlama packet is target-useful
  for a Qwen score receiver under destructive packet controls.
- The scout quantifies remaining receiver headroom: target-or-packet oracle is
  materially above packet-only.
- The current method is a byte/exposure-distinct task packet, not prefix
  tuning, prompt tuning, or KV cache sharing.

Unsafe:

- Claiming the receiver learned a universal latent language.
- Claiming the selected receiver beats the source packet itself.
- Claiming native systems superiority over C2C, KVComm, vLLM, SGLang, KIVI,
  KVQuant, QJL, or TurboQuant.
