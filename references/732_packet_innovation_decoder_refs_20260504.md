# Packet-Innovation Decoder Reference Refresh

Date: 2026-05-04

## Why This Diagnostic Matters

The behavior-atom decoder partially improved Qwen3 on a tiny strict
ARC-Challenge disagreement slice, but zero-source and same-source-choice
wrong-row controls stayed too competitive. The packet-innovation diagnostic
therefore removes the receiver intercept and only permits packet-dependent
candidate residuals. A zero packet must decode to zero.

The diagnostic fails: removing target-only receiver bias also removes the
held-out positive lift. That makes the next branch a receiver/gating problem,
not a reason to widen benchmarks.

## Dense Communication Competitors

- [Cache-to-Cache: Direct Semantic Communication Between Large Language
  Models](https://openreview.net/forum?id=LeatkxrBCi) is the main dense
  baseline. It projects and fuses source KV cache into the target cache, with a
  learned layer gate, and reports accuracy and latency gains over text
  communication. SRP should treat C2C as the high-bandwidth baseline and avoid
  claiming latency superiority until measured natively.
- [KVComm: Enabling Efficient LLM Communication through Selective KV
  Sharing](https://openreview.net/forum?id=F7rUng23nw) selectively transmits
  informative KV pairs and reports near upper-bound performance while sending a
  fraction of layers' KV pairs. SRP differs only if it proves fixed-byte,
  source-private packets can improve utility without exposing source KV/hidden
  state.
- [Communicating Activations Between Language Model
  Agents](https://arxiv.org/abs/2501.14082) and latent-MAS/latent relay
  systems are high-bandwidth or dense-state communication competitors. They
  sharpen the need for SRP to claim a distinct byte/exposure operating point.

## Sparse Basis And Receiver Direction

- [Sparse Autoencoders Find Highly Interpretable Features in Language
  Models](https://arxiv.org/abs/2309.08600), [BatchTopK Sparse
  Autoencoders](https://arxiv.org/abs/2412.06410), and [SAE feature
  universality](https://arxiv.org/abs/2410.06981) make sparse/common feature
  bases plausible but not novel by themselves.
- [Sparse Crosscoders for Cross-Layer Features and Model
  Diffing](https://transformer-circuits.pub/2024/crosscoders/index.html) and
  [Cross-Architecture Model Diffing with
  Crosscoders](https://arxiv.org/abs/2602.11729) support shared/private feature
  partitions. The most relevant next SRP branch is a behavior-loss DFC-style
  packet atom bank that separates shared, source-private, and target-private
  features.
- [Transcoders Find Interpretable LLM Feature
  Circuits](https://arxiv.org/abs/2406.11944) motivates behavior-oriented
  sparse replacements. For SRP, the feature objective should predict target
  correction utility, not only reconstruct source activations.

## Quantization And Systems Boundary

- [TurboQuant](https://arxiv.org/abs/2504.19874) is now a strong low-bit vector
  and KV quantization comparator, with near-optimal distortion claims and KV
  quality neutrality around low bit rates. SRP should not claim generic
  cache-compression superiority.
- [KVQuant](https://arxiv.org/abs/2401.18079), [QJL
  quantization](https://arxiv.org/abs/2406.03482), and
  [KIVI](https://arxiv.org/abs/2402.02750) set strong KV byte/memory floors.
  SRP can report payload/framed bytes and source exposure, but native systems
  wins require GPU/vLLM/SGLang measurements.
- [vLLM/PagedAttention](https://arxiv.org/abs/2309.06180) and
  [SGLang/RadixAttention](https://arxiv.org/abs/2312.07104) are the right
  serving surfaces for later native systems rows.

## Benchmark And Control Boundary

- ARC-Challenge, OpenBookQA, and HellaSwag remain the right benchmark ladder:
  ARC is the Mac-local decision surface, OpenBookQA is the nearest science-QA
  generalization, and HellaSwag stresses candidate/text-order artifacts.
- Same-source-choice wrong-row, target-derived packet, zero packet,
  atom/coefficient shuffle, top-atom knockout, candidate roll/derangement,
  source-index/rank/score, source-score quantization, same-byte visible text,
  and Qwen substitution remain mandatory before any positive claim.
- The packet-innovation row proves that a zero-source intercept fix is
  insufficient. The next gate must learn a safe act/no-op decision under
  corruptions rather than merely adding more linear packet features.

## Next Promoted Gate

Promote an event-triggered accept/abstain innovation receiver:

1. matched packets are trained as possible act events;
2. zero, wrong-row, atom-shuffled, coefficient-shuffled, candidate-rolled, and
   source-choice-matched packets are trained as hold/no-op or explicit harm
   examples;
3. abstention returns target-only exactly;
4. pass condition requires matched packets to beat every required shortcut and
   destructive control with positive paired uncertainty.

If that fails, promote a tiny DFC/BatchTopK behavior atom preflight that
transmits only source-private innovation atoms and uses the same act/no-op
control suite.

## Takeaway

The literature makes dense KV/activation communication and sparse feature
bases credible prior art. The remaining LatentWire novelty is narrower and
testable: fixed-byte, quantized, source-private packets that improve a receiver
only when row-specific source evidence is present. The packet-innovation
diagnostic did not prove this, but it cleanly rules out the linear no-intercept
decoder as sufficient.
