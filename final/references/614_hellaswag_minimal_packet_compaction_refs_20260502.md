# References: HellaSwag Minimal Packet Compaction

## Claim Boundary

This memo supports the minimal-packet compaction artifact. The safe claim is
that the promoted HellaSwag hidden-innovation packet rows do not need the
confidence/debug byte at runtime: a one-byte candidate-id packet exactly
reproduces the selected predictions while reducing logical raw and framed
payload bytes. It does not claim solved cross-model latent reasoning, native
serving speedup, HBM traffic reduction, or superiority to cache/KV compression
methods.

## Primary Sources And Why They Matter

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - https://arxiv.org/abs/1905.07830
   - Why it matters: frozen full-validation decision surface for the promoted
     packet rows.

2. Cache-to-Cache and KV Communication
   - https://openreview.net/forum?id=LeatkxrBCi
   - https://openreview.net/forum?id=yGOytgjurF
   - Boundary: these methods communicate through KV/cache states. The compact
     LatentWire row sends only a task-level candidate-id packet and no source
     KV/cache state.

3. QJL, TurboQuant, KIVI, and KVQuant
   - https://arxiv.org/abs/2504.19874
   - https://arxiv.org/abs/2402.02750
   - https://arxiv.org/abs/2401.18079
   - Boundary: these are vector/KV-cache compression methods. The compaction
     artifact is not preserving vector fidelity; it is a task-level rate point
     for source-private decision communication.

4. vLLM/PagedAttention and SGLang
   - https://arxiv.org/abs/2309.06180
   - https://arxiv.org/abs/2312.07104
   - Boundary: native serving throughput and HBM traffic claims remain disabled
     until NVIDIA/vLLM/SGLang rows are run.

5. Prefix-Tuning and P-Tuning v2
   - https://aclanthology.org/2021.acl-long.353/
   - https://arxiv.org/abs/2110.07602
   - Boundary: prefix/prompt tuning learns persistent continuous conditioning
     vectors. The compact packet is a per-example discrete candidate id, not a
     soft prompt or prefix token sequence.

6. BLIP-2 Q-Former, Flamingo, and Perceiver IO
   - https://arxiv.org/abs/2301.12597
   - https://arxiv.org/abs/2204.14198
   - https://arxiv.org/abs/2107.14795
   - Boundary: learned query/resampler connectors remain a future GPU branch.
     This artifact is a byte-level compaction of the existing packet.

## Reviewer-Facing Framing

Safe:

- The current HellaSwag decoder only needs the selected candidate id at
  runtime.
- A one-byte packet exactly reproduces all promoted Qwen and TinyLlama packet
  predictions.
- The systems row improves from `2B` raw / `5B` framed to `1B` raw / `4B`
  framed, with no source text, KV, raw hidden vector, or raw score vector.

Unsafe:

- Claiming receiver/common-language progress.
- Claiming GPU throughput, HBM traffic, or kernel wins.
- Claiming this subsumes C2C, KVComm, QJL/TurboQuant, KIVI, or KVQuant.
- Claiming this is prefix tuning or a learned query connector.
