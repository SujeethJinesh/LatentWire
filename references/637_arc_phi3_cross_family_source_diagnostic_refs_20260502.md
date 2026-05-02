# ARC Phi-3 Cross-Family Source Diagnostic References

Date: 2026-05-02

## Local Evidence

- Gate:
  `results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu/source_family_cache_falsification.json`
- Decision: Phi-3 is a negative cross-family source diagnostic for the current
  ARC Fourier/anchor-syndrome packet.
- Frozen test full-slice matched/target/text: `0.244/0.265/0.241`.
- Frozen test Qwen-disagreement matched/Qwen-substituted/text/target:
  `0.200/0.340/0.209/0.273`.
- Frozen test minimum matched-minus-Qwen-substituted and CI95 low:
  `-0.143`, `-0.193`.
- Phi source-choice accuracy before packet: validation `0.274`, test `0.246`.

## Related Work Boundary

- Relative representations. Use to frame the shared-coordinate packet as an
  anchor/common-coordinate method rather than raw hidden transfer:
  `https://arxiv.org/abs/2209.15430`.
- C2C. Boundary: C2C projects and fuses source KV caches; LatentWire sends a
  fixed 12-byte source-private packet and exposes no source KV state:
  `https://arxiv.org/abs/2510.03215`.
- KVComm. Boundary: selective KV sharing transmits a chosen subset of KV pairs,
  which is the correct high-rate baseline for systems comparison:
  `https://openreview.net/forum?id=F7rUng23nw`.
- KVCOMM. Boundary: online cross-context KV reuse aligns cache offsets across
  agents; it is a cache-reuse systems baseline, not a fixed per-example packet:
  `https://arxiv.org/abs/2510.12872`.
- Latent K-V cache alignment and activation communication. These are closer
  latent-transfer competitors and motivate why reviewer-facing comparisons
  must include source-state exposure and model-internal requirements:
  `https://arxiv.org/abs/2601.06123`,
  `https://arxiv.org/abs/2501.14082`.
- Sparse crosscoders and universal SAEs. These motivate the next hidden/query
  common-basis branch, but the current Phi-3 row does not learn a shared SAE or
  crosscoder dictionary:
  `https://transformer-circuits.pub/2024/crosscoders/index.html`,
  `https://arxiv.org/abs/2502.03714`.
- BLIP-2/Q-Former. Query bottlenecks are a plausible connector design pattern;
  they are not duplicated by the current cached source-choice packet:
  `https://arxiv.org/abs/2301.12597`.
- TurboQuant, KIVI, KVQuant, and QJL. These define KV/cache quantization and
  byte-floor baselines for the systems section, not direct semantic packet
  transfer methods:
  `https://arxiv.org/abs/2504.19874`,
  `https://arxiv.org/abs/2402.02750`,
  `https://arxiv.org/abs/2401.18079`,
  `https://arxiv.org/abs/2406.03482`.
- Prefix tuning. Boundary: learned virtual prompts are persistent prompt
  parameters, not per-example source-private packets:
  `https://arxiv.org/abs/2101.00190`.

## Paper Implication

This row strengthens the falsification ladder. It says the current ARC
Fourier/anchor-syndrome result should be framed as common-basis packet transfer
with Qwen-family source evidence, not solved cross-family latent
communication. The next positive method needs either a stronger non-Qwen source
or a richer hidden/query connector with the same byte/exposure accounting and
destructive controls.
