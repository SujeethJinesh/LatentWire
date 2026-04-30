# Packet Trace Card v2 Systems References

- date: `2026-04-30`
- role: primary-source systems memo for source-private packet traffic accounting
  and local Mac transport microbench framing

## Sources and Experiment Implications

1. **TurboQuant**
   Source: https://arxiv.org/abs/2504.19874
   Blocker it helps with: reviewers may argue modern KV quantization erases the
   packet byte advantage.
   Mechanism/design idea: compare against quantized residual/KV-style byte
   floors, but keep task boundaries separate.
   Next experiment change: keep TurboQuant as a KV/cache compression baseline
   and a future native GPU comparator, not as a direct source-private packet
   baseline.
   Role: systems baseline / framing.

2. **QJL / Quantized Johnson-Lindenstrauss Compression**
   Source: https://arxiv.org/abs/2406.03482
   Blocker it helps with: need a strong 1-bit projection/residual lower-bound
   comparator.
   Mechanism/design idea: include 1-bit source-KV byte floors and use QJL-style
   residual coding as inspiration for future learned packets.
   Next experiment change: report QJL byte-floor rows and keep source-private
   packet claims separate from KV-state transport.
   Role: systems baseline / method inspiration.

3. **KIVI**
   Source: https://arxiv.org/abs/2402.02750
   Blocker it helps with: reviewers expect 2-bit KV cache quantization baselines.
   Mechanism/design idea: K/V caches have structured quantization axes and are
   served as internal model state, not endpoint side-information packets.
   Next experiment change: keep KIVI/KVQuant as KV-cache byte-floor and future
   NVIDIA serving baselines.
   Role: systems baseline.

4. **KVQuant**
   Source: https://arxiv.org/abs/2401.18079
   Blocker it helps with: sub-4-bit KV cache compression may look like a direct
   competitor.
   Mechanism/design idea: compare private packet transport against compressed
   KV-state movement only under explicit access assumptions.
   Next experiment change: add native CUDA/KV rows only when server hardware is
   available.
   Role: systems baseline.

5. **vLLM / PagedAttention**
   Source: https://arxiv.org/abs/2309.06180
   Blocker it helps with: serving reviewers care about batching, KV memory
   management, and throughput rather than raw payload bytes alone.
   Mechanism/design idea: report packet batch amortization and future goodput
   under TTFT/TPOT constraints.
   Next experiment change: use vLLM-compatible serving telemetry when NVIDIA
   hardware is available.
   Role: systems framing / future baseline.

6. **FlashAttention**
   Source: https://arxiv.org/abs/2205.14135
   Blocker it helps with: hardware systems claims need IO-aware accounting.
   Mechanism/design idea: separate raw semantic payload from cache-line, DMA,
   HBM/SRAM, and model-compute movement.
   Next experiment change: keep packet trace card explicit about transfer
   quanta and non-claims.
   Role: systems framing.

7. **DistServe**
   Source: https://arxiv.org/abs/2401.09670 and
   https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
   Blocker it helps with: endpoint latency should be stated in TTFT/TPOT/goodput
   language, not vague "faster" claims.
   Mechanism/design idea: future server trace card should report TTFT, TPOT,
   goodput, SLO attainment, and prefill/decode separation where relevant.
   Next experiment change: Mac proxy rows remain local transport evidence;
   production claims wait for server telemetry.
   Role: systems framing / future baseline.

8. **Apple M1 Pro / M1 Max unified memory context**
   Source:
   https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/
   Blocker it helps with: local Mac results need hardware scope.
   Mechanism/design idea: frame local measurements as unified-memory transport
   microbenchmarks, not accelerator serving results.
   Next experiment change: keep Mac packet-ring microbench as a local systems
   trace and explicitly mark remote GPU telemetry as future work.
   Role: hardware context / limitation framing.
