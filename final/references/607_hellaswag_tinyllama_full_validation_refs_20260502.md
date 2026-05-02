# References: TinyLlama Full-Validation Source-Family Gate

## Claim Boundary

The new evidence supports a source-family robustness claim for the fixed-byte hidden-innovation packet. It does not establish receiver-family transfer, a universal latent basis, or superiority over cache/KV communication systems.

## Primary Sources And Why They Matter

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - https://arxiv.org/abs/1905.07830
   - Why it matters: benchmark surface used for the full validation gate.

2. TinyLlama: An Open-Source Small Language Model
   - https://arxiv.org/abs/2401.02385
   - Why it matters: non-Qwen source family used in the full-validation stress run.

3. Prefix-Tuning: Optimizing Continuous Prompts for Generation
   - https://aclanthology.org/2021.acl-long.353/
   - Boundary: prefix tuning learns continuous target-conditioning vectors; LatentWire transmits a fixed `2B` raw / `5B` framed discrete packet and does not prepend learned continuous vectors at inference.

4. The Power of Scale for Parameter-Efficient Prompt Tuning
   - https://arxiv.org/abs/2104.08691
   - Boundary: prompt tuning learns task prompts for one model; this gate tests source-private model-to-model evidence under destructive controls.

5. LoRA: Low-Rank Adaptation of Large Language Models
   - https://openreview.net/forum?id=nZeVKeeFYf9
   - Boundary: LoRA changes model weights through trainable low-rank adapters; LatentWire keeps the base source and receiver contract frozen at evaluation.

6. Cache-to-Cache: Direct Semantic Communication Between Large Language Models
   - https://arxiv.org/abs/2510.03215
   - Boundary: C2C fuses projected KV/cache state. LatentWire does not transmit source KV, raw hidden vectors, raw scores, or text.

7. KVComm: Enabling Efficient LLM Communication via Selective KV Cache Sharing
   - https://arxiv.org/abs/2510.03346
   - Boundary: KVComm is a KV-sharing communication baseline; LatentWire's systems row is byte/exposure accounting, not a defeated native KVComm row.

8. Relative Representations Enable Zero-Shot Latent Space Communication
   - https://arxiv.org/abs/2209.15430
   - Boundary: relative representations use anchor-relative coordinates for latent communication. The current TinyLlama result does not claim an invariant shared coordinate system.

9. Sparse Autoencoders Find Highly Interpretable Features in Language Models
   - https://arxiv.org/abs/2309.08600
   - Boundary: SAE work motivates sparse/common-basis future branches; the current method is not an SAE or crosscoder feature-basis method.

10. Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models
    - https://arxiv.org/abs/2410.06981
    - Boundary: universal sparse feature spaces are relevant to the next common-language branch, but the full TinyLlama pass only proves source-family robustness under this packet contract.

11. Crosscoders
    - https://transformer-circuits.pub/2025/crosscoder-diffing/index.html
    - Boundary: crosscoders separate shared and model-specific features; LatentWire has not yet trained an interpretable shared sparse dictionary for the packet.

12. QJL, KIVI, KVQuant, and TurboQuant
    - https://arxiv.org/abs/2406.03482
    - https://arxiv.org/abs/2402.02750
    - https://arxiv.org/abs/2401.18079
    - https://arxiv.org/abs/2504.19874
    - Boundary: these are KV/cache compression or sketching references. They are systems comparators for bytes and memory traffic, not evidence that LatentWire beats native KV compression.

13. vLLM and SGLang
    - https://arxiv.org/abs/2309.06180
    - https://arxiv.org/abs/2312.07104
    - Boundary: production serving baselines remain future NVIDIA work. The current systems card is Mac-local only.

## Reviewer-Facing Framing

Safe:

- The fixed-byte source-private packet is not only a Qwen-source artifact; it also passes full HellaSwag validation with TinyLlama as the source model.
- The packet provides a byte/exposure-distinct alternative to text, prefix, and KV-state communication.
- The Mac run demonstrates feasibility and records phase timing, but not production serving throughput.

Unsafe:

- General cross-family latent communication.
- Universal model-to-model latent language.
- Beating C2C, KVComm, or KV quantization on their native systems axes.
- Native GPU or HBM traffic claims.
