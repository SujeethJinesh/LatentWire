# Systems Novelty Case for Source-Private Packet + Frozen Verifier

- date: `2026-04-30`
- role: primary-source systems/literature scout for the source-private
  2-byte packet plus frozen target verifier branch
- current status: useful ICLR systems framing, but not broad latent-transfer
  readiness

## Current Gate

The live positive row is a source-private diagnostic packet consumed by a
frozen target verifier with public candidate side information. The strongest
current systems evidence is the packet trace card and Mac packet-ring
microbench; the remaining reviewer risk is whether this is novel beyond KV
compression/cache transfer and whether the frozen verifier cost is honestly
measured.

## Sources and Implications

1. **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
   - Source: https://arxiv.org/abs/2504.19874
   - Blocker helped: modern vector/KV quantization weakens naive fp16 KV byte
     floors.
   - Mechanism idea: randomized rotation, optimal scalar quantization, and a
     QJL residual for inner-product preservation.
   - Next experiment change: keep TurboQuant as a KV-state comparator and
     protected-residual inspiration, not as evidence against source-private
     packet novelty.
   - Role: baseline / framing.

2. **QJL: 1-Bit Quantized JL Transform for KV Cache Quantization**
   - Source: https://arxiv.org/abs/2406.03482
   - Blocker helped: 1-bit sketches are the strongest compact-state lower-bound
     pressure on a 2-byte packet claim.
   - Mechanism idea: Johnson-Lindenstrauss projection plus sign quantization
     with an asymmetric unbiased inner-product estimator.
   - Next experiment change: add or keep QJL-style source-KV byte floors and,
     for future learned packets, include random-projection/sign-sketch controls
     under the same destructive source gates.
   - Role: baseline / ablation inspiration.

3. **KIVI and KVQuant**
   - Sources: https://arxiv.org/abs/2402.02750 and
     https://arxiv.org/abs/2401.18079
   - Blocker helped: reviewers will expect practical sub-4-bit KV-cache
     baselines, not only theoretical or fp16 comparisons.
   - Mechanism idea: KIVI uses asymmetric 2-bit K/V quantization; KVQuant uses
     per-channel/pre-RoPE/outlier-aware low-bit KV quantization.
   - Next experiment change: systems tables should mark these as compressed
     source-KV-state regimes with explicit exposure assumptions; do not claim
     LatentWire wins native KV-cache serving tasks.
   - Role: systems baseline.

4. **Cache-to-Cache and KVCOMM**
   - Sources: https://arxiv.org/abs/2510.03215 and
     https://arxiv.org/abs/2510.12872
   - Blocker helped: closest competitors to broad "LLMs communicate without
     text" language.
   - Mechanism idea: C2C projects/fuses source KV cache into target cache;
     KVCOMM reuses and offset-aligns cache state across multi-agent contexts.
   - Next experiment change: frame LatentWire as source-private task evidence
     with public side information, not as cache fusion or cache reuse. Keep
     cache-transfer rows as a distinct high-rate, source-state-exposed family.
   - Role: competitor / framing.

5. **vLLM / PagedAttention**
   - Source: https://arxiv.org/abs/2309.06180
   - Blocker helped: serving reviewers care about KV block management,
     batching, sharing, and throughput, not only payload bytes.
   - Mechanism idea: page KV cache into blocks and schedule requests around
     memory fragmentation and sharing.
   - Next experiment change: future GPU run should be vLLM-shaped: batch sweep,
     TTFT, TPOT, goodput, prompt/generated tokens, and KV bytes.
   - Role: serving baseline / future harness target.

6. **DistServe**
   - Sources: https://arxiv.org/abs/2401.09670 and
     https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
   - Blocker helped: local p50 latency is not enough for systems credibility.
   - Mechanism idea: separate prefill and decode, then optimize serving
     goodput under TTFT/TPOT SLOs.
   - Next experiment change: keep Mac results as proxy evidence; any production
     claim needs TTFT/TPOT/goodput/SLO metrics.
   - Role: systems metric standard.

7. **FlashAttention**
   - Source: https://arxiv.org/abs/2205.14135
   - Blocker helped: a 2-byte semantic packet is not literally a 2-byte
     hardware transfer.
   - Mechanism idea: IO-aware accounting across memory hierarchy, especially
     HBM/SRAM reads and writes.
   - Next experiment change: preserve raw bytes, 64B line bytes, 128B burst
     bytes, batch amortization, and explicit non-claims in every systems table.
   - Role: systems framing / overclaim guard.

8. **GainSight / Tambe-style hardware framing**
   - Sources: https://arxiv.org/abs/2504.14866 and
     https://profiles.stanford.edu/thierry-tambe
   - Blocker helped: a hardware-facing contribution must discuss data lifetime,
     locality, accelerator memory, and algorithm-hardware interfaces.
   - Mechanism idea: profile lifetime and movement of transient data rather
     than only counting abstract bits.
   - Next experiment change: add a verifier-consumption trace that reports
     packet lifetime, prompt-token lifetime, target forward-pass count, RSS,
     and whether source text/KV crosses the boundary.
   - Role: hardware framing / ablation design.

9. **Diffusion/Consistency/Flow receiver inspiration**
   - Sources: https://arxiv.org/abs/2212.09748,
     https://arxiv.org/abs/2303.01469, and https://arxiv.org/abs/2210.02747
   - Blocker helped: the frozen verifier works but remains protocol-shaped;
     a stronger method branch needs a less hand-coded receiver.
   - Mechanism idea: DiT suggests tokenized transformer denoising; consistency
     models suggest one-step/few-step endpoint maps; flow matching suggests a
     learned path from target prior logits to source-conditioned posterior.
   - Next experiment change: only pursue a tiny candidate-logit denoiser/flow
     after the binary verifier clears cross-family/n160. It must include
     source-destroying views that map back to target prior.
   - Role: inspiration / objective design.

## Concrete Mac-Local Systems Ablation

Run `source_private_verifier_consumption_trace_202605xx` on CPU/MPS-free Mac:

- rows: `target_only`, `2B_packet_binary_verifier`, `query_aware_text`,
  `full_hidden_log`, `random_same_byte`, `shuffled_packet`;
- fixed slice: start `n=32`, then `n=64` if runtime is acceptable;
- metrics: accuracy, valid rate, paired delta vs target, prompt tokens,
  target forward passes per example, p50/p95 wall time, peak RSS, raw/line/DMA
  source-boundary bytes, prompt-KV byte floor, and source exposure flags;
- control: same frozen Qwen binary verifier and same public candidate set for
  all rows.

This improves on the packet trace card by measuring the cost of consuming the
packet with the frozen verifier, not just transporting the boundary object. It
also makes the main systems caveat explicit: LatentWire wins source-boundary
privacy/rate, while the current frozen verifier spends target-side compute.

## Decision

- Saturated: byte-only packet claims and naive KV fp16 floors.
- Weakened: any broad claim that LatentWire beats KV/cache methods on native
  serving throughput.
- Alive: frozen binary verifier if it clears cross-family and n160 controls.
- Promoted: verifier-consumption trace as the next Mac-local systems ablation.

