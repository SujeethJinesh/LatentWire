# Mac Unified-Memory Transport Profile References

- date: `2026-04-30`
- purpose: primary-source grounding for the Mac-local unified-memory transport
  profile and its systems claim boundaries.

## Apple M1 Max / Unified-Memory Hardware

- source: https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/
- blocker helped: the paper needs a hardware-readable systems story while the
  current evidence is Mac-local, not NVIDIA/HBM.
- mechanism idea: report a unified-memory boundary trace-card instead of
  pretending a two-byte semantic payload is literally a two-byte hardware
  transfer.
- next experiment change: add host-profile metadata and keep cache-line/DMA
  rounding in the Mac transport profile.
- role: systems framing / hardware context.

## MLX Unified Memory

- source: https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
- blocker helped: reviewers may ask what "Mac unified memory" changes relative
  to host-device staging.
- mechanism idea: unified memory changes access/staging assumptions but not the
  fact that bytes, cache lines, synchronization, and backend fallback still
  matter.
- next experiment change: explicitly label the profile as boundary accounting,
  not measured accelerator throughput.
- role: systems framing / overclaim guard.

## Apple PyTorch MPS

- source: https://developer.apple.com/metal/pytorch/
- blocker helped: the local machine can run PyTorch/MPS, but previous Qwen3
  MPS probing hit backend limitations.
- mechanism idea: keep the profile CPU-only over existing endpoint rows until
  an MPS generation path is stable.
- next experiment change: do not start a new MPS generation job for this gate;
  use artifact accounting and record the CPU-only execution note.
- role: implementation constraint / systems caveat.

## PyTorch MPS Profiling Controls

- source: https://docs.pytorch.org/docs/stable/mps_environment_variables.html
- blocker helped: future Mac-native telemetry needs an explicit MPS profiler /
  fallback setup rather than ad hoc timing.
- mechanism idea: separate deterministic accounting now from MPS-profiler
  telemetry later.
- next experiment change: leave native MPS counters as a future gate after the
  current profile, not as part of this pass/fail rule.
- role: future telemetry path.

## FlashAttention

- source: https://arxiv.org/abs/2205.14135
- blocker helped: systems reviewers will not accept raw byte counts alone if
  memory hierarchy and IO are ignored.
- mechanism idea: report IO-style movement, transfer quanta, and memory
  hierarchy proxies next to accuracy.
- next experiment change: include raw payload bytes, 64B line rounding, 128B
  DMA rounding, and batch-packing fields.
- role: systems framing / IO-awareness.

## vLLM / PagedAttention

- source: https://arxiv.org/abs/2309.06180
- blocker helped: packet communication must be compared carefully against
  serving systems built around KV-cache memory management.
- mechanism idea: distinguish source-private evidence packets from KV/cache
  transport and mark `source_kv_exposed` explicitly.
- next experiment change: keep KV byte-floor rows and non-claims; do not claim
  native superiority over vLLM/PagedAttention.
- role: serving baseline / comparison boundary.

## DistServe

- source: https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
- blocker helped: reviewers expect TTFT/TPOT/goodput terminology for serving
  claims.
- mechanism idea: report TTFT proxy values but reserve TPOT/goodput for native
  serving runs.
- next experiment change: this profile keeps TTFT/E2E from endpoint rows and
  explicitly avoids production throughput claims.
- role: metric framing / future NVIDIA gate.

## KIVI, TurboQuant, QJL

- sources:
  - https://arxiv.org/abs/2402.02750
  - https://arxiv.org/abs/2504.19874
  - https://arxiv.org/abs/2206.14894
- blocker helped: low-bit KV/vector compression is a strong neighboring
  baseline and makes generic "we compress better" claims unsafe.
- mechanism idea: compare against KV byte floors under one-bit, two-bit, and
  TurboQuant-style bit-width assumptions, while preserving the access-model
  distinction.
- next experiment change: use the existing Qwen3 geometry to compute prompt-KV
  byte deltas, but mark them as accounting rows rather than native kernels.
- role: compression baseline / claim boundary.

## C2C / KV Communication

- source: https://arxiv.org/abs/2510.03215
- blocker helped: the closest cross-model systems competitor is high-rate
  internal-state communication, not only text relay.
- mechanism idea: contrast source-private endpoint packets with cache/KV fusion
  by access rights, byte floor, and private-state exposure.
- next experiment change: keep C2C/KVComm as a high-rate baseline family for
  future GPU runs; do not claim this Mac profile beats it natively.
- role: competitor boundary.

## Bottom Line

The profile strengthens the systems contribution because it is honest about
transfer granularity. The paper can claim that LatentWire occupies the far-left
source-private payload point and avoids source text/KV movement; it should not
claim production throughput or native KV-compression superiority until a real
serving stack with hardware counters is available.
