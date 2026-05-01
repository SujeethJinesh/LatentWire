# Cross-Benchmark Systems Comparator References

Web check: 2026-05-01. Scope: primary sources for the systems comparator
against cache/KV communication and quantized source-state baselines.

## Local Result

Artifact:
`results/source_private_cross_benchmark_systems_comparator_20260501/`

- pass gate: `True`
- headline-eligible benchmarks: `2` (`ARC-Challenge`, `OpenBookQA`)
- diagnostic benchmarks: `1` (`HellaSwag`, marked non-headline due to
  source-label-copy threat)
- framed packet range: `5-15B`
- QJL-style 1-bit one-source-token KV floor versus framed packet: at least
  `51.2x`
- 30%-layer QJL-style floor versus framed packet: at least `15.36x`
- 30%-layer fp16 KVComm-style floor versus framed packet: at least `245.76x`
- TurboQuant-style 3.5-bit source-state floor versus framed packet: at least
  `179.2x`

These are state-exposure byte floors from the local Qwen2.5-0.5B config, not
native quality or throughput comparisons.

## Cache And KV Communication Baselines

- Fu et al., Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models, ICLR 2026. C2C projects and fuses source KV caches into a
  target model and reports latency gains versus text communication. This is
  the closest semantic-cache baseline, but its access model exposes source
  cache state. https://arxiv.org/abs/2510.03215
- Shi et al., KVComm: Enabling Efficient LLM Communication through Selective
  KV Sharing, ICLR 2026. KVComm selects informative KV pairs/layers and reports
  as few as `30%` of layers' KV pairs in its abstract. The local comparator uses
  that `30%` only as a byte-floor assumption until we run KVComm natively.
  https://arxiv.org/abs/2510.03346
- Ye et al., KVCOMM: Online Cross-context KV-cache Communication for Efficient
  LLM-based Multi-agent Systems, 2025. KVCOMM is a cross-context KV-cache
  communication/pre-fill reuse neighbor; fair comparison must report source-KV
  exposure and native serving metrics. https://arxiv.org/abs/2510.12872

## Quantization And Serving Baselines

- Zandieh et al., QJL: 1-Bit Quantized JL Transform for KV Cache Quantization
  with Zero Overhead, 2024. QJL motivates the one-bit sign-sketch byte floor,
  but a real baseline would need a quality/latency run, not only byte math.
  https://arxiv.org/abs/2406.03482
- Zandieh et al., TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate, 2025. TurboQuant motivates the vector/KV low-bit floor and
  future packet-sketch ablations; the local row is only a bits-per-element
  proxy. https://arxiv.org/abs/2504.19874
- Kwon et al., Efficient Memory Management for Large Language Model Serving
  with PagedAttention, SOSP 2023. vLLM/PagedAttention defines the native serving
  substrate where TTFT, TPOT, goodput, memory, and KV-cache measurements should
  be run once NVIDIA hardware is available. https://arxiv.org/abs/2309.06180

## Safe Boundary

The systems claim strengthened here is narrow: source-private packets occupy a
5-15B framed-record regime on positive ARC/OpenBookQA rows while a conservative
one-token source-state floor is already tens to hundreds of times larger. The
paper must not claim native throughput superiority over C2C, KVComm/KVCOMM,
QJL, TurboQuant, or vLLM until those baselines are run in their native setting.
