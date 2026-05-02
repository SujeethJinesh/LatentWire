# References: HellaSwag Sparse-Query Receiver

## Claim Boundary

This memo supports the train-only sparse-query receiver artifact. The safe
claim is that a full-hidden low-rank query receiver was tested and failed to
clear the receiver-improvement gate. It does not claim solved cross-model
latent reasoning, first latent communication, or systems superiority over
cache-transfer and KV-quantization methods.

## Primary Sources And Why They Matter

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - https://arxiv.org/abs/1905.07830
   - Why it matters: the frozen full-validation benchmark surface.

2. Relative Representations Enable Zero-Shot Latent Space Communication
   - https://arxiv.org/abs/2209.15430
   - Boundary: relative representations use anchor-relative coordinates to
     stabilize latent comparisons. This artifact uses train-only low-rank
     receiver queries over task features and fixed-byte packets, not dense
     latent-space communication.

3. Sparse Autoencoders, Universal Feature Spaces, and Sparse Crosscoders
   - https://arxiv.org/abs/2309.08600
   - https://arxiv.org/abs/2410.06981
   - https://transformer-circuits.pub/2024/crosscoders/index.html
   - Boundary: these motivate sparse/common-basis hypotheses. This artifact
     does not train a full SAE/crosscoder or reconstruct model activations; it
     tests low-rank receiver-query features for packet arbitration.

4. BLIP-2 Q-Former, Flamingo, and Perceiver IO
   - https://arxiv.org/abs/2301.12597
   - https://arxiv.org/abs/2204.14198
   - https://arxiv.org/abs/2107.14795
   - Why it matters: learned query bottlenecks are the architectural precedent
     for future connector work. This artifact is only a Mac-feasible low-rank
     receiver-query probe, not a full cross-attention connector.

5. Prefix-Tuning and P-Tuning v2
   - https://aclanthology.org/2021.acl-long.353/
   - https://arxiv.org/abs/2110.07602
   - Boundary: prefix/prompt methods learn persistent virtual-token
     conditioning. LatentWire's packet is per-example, fixed-byte, and does
     not inject soft tokens or train receiver prompts.

6. Cache-to-Cache and KV Communication
   - https://openreview.net/forum?id=LeatkxrBCi
   - https://arxiv.org/abs/2510.03346
   - https://arxiv.org/abs/2510.12872
   - https://arxiv.org/abs/2512.17914
   - Boundary: C2C/KVComm/KVCOMM/Q-KVComm communicate through projected,
     shared, reused, or compressed KV/cache state. This artifact sends no
     source KV, source raw hidden vector, source raw score vector, or source
     text.

7. QJL, KIVI, KVQuant, and TurboQuant
   - https://arxiv.org/abs/2406.03482
   - https://arxiv.org/abs/2402.02750
   - https://arxiv.org/abs/2401.18079
   - https://arxiv.org/abs/2504.19874
   - Boundary: these are KV/vector compression methods. They motivate
     rate-distortion comparisons, but this artifact is task-level packet
     arbitration rather than vector-fidelity compression.

8. Diffusion Transformers and Latent Diffusion
   - https://arxiv.org/abs/2212.09748
   - https://arxiv.org/abs/2112.10752
   - Boundary: diffusion-style latent repair remains a possible future method
     branch. The current artifact is one-shot receiver selection.

9. vLLM / PagedAttention and SGLang
   - https://arxiv.org/abs/2309.06180
   - https://arxiv.org/abs/2312.07104
   - Boundary: native serving comparisons remain pending until NVIDIA GPU rows
     are available.

## Reviewer-Facing Framing

Safe:

- The artifact tests whether low-rank full-hidden receiver queries can close
  TinyLlama/Qwen complementary-error headroom.
- The predeclared row fails, and the best scout is matched by random-query and
  hidden-row-shuffle controls.
- The result narrows the next branch away from receiver selectors and toward a
  learned source packet/code, larger calibration, or a true joint connector.

Unsafe:

- Claiming receiver improvement over packet-only.
- Claiming the sparse-query branch learned a common language.
- Claiming novelty over all latent communication, prefix tuning, C2C, KVComm,
  or KV quantization.
- Claiming native systems superiority before GPU/vLLM/SGLang rows.
