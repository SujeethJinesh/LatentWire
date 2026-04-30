# Memory Traffic Ledger References

- date: `2026-04-30`
- purpose: primary-source grounding for the source-private memory traffic ledger
  and packet-ISA trace-card framing.

## Tambe Lab Research Themes

- source: https://tambelab.stanford.edu/
- blocker helped: the paper needs a systems story that is legible to
  hardware/accelerator reviewers, not just an accuracy table.
- mechanism idea: frame LatentWire as an explicit algorithm-hardware interface:
  typed source-private packet fields, receiver contract, traffic lifetime, and
  forbidden private-state movement.
- next experiment change: add a packet trace card that records memory movement
  and contract fields together.
- role: paper framing / systems motivation.

## FlashAttention

- source: https://arxiv.org/abs/2205.14135
- blocker helped: byte-only claims can miss the real systems issue, because
  transformer runtime is often governed by IO/memory movement rather than FLOPs
  alone.
- mechanism idea: report cache-line and DMA-burst traffic proxies alongside raw
  semantic payload.
- next experiment change: keep line/burst rounding in the systems table and do
  not claim a literal 2-byte hardware transfer.
- role: systems framing / overclaim guard.

## vLLM / PagedAttention

- source: https://arxiv.org/abs/2309.06180
- blocker helped: reviewers will compare packet communication with serving
  systems whose dominant state is KV cache memory.
- mechanism idea: keep KV byte-floor rows as explicit cache-management
  comparators and separate them from native packet rows.
- next experiment change: record whether a method exposes or transports source
  KV/cache tensors in every systems row.
- role: baseline framing / systems comparison.

## GainSight / Heterogeneous On-Chip Memory Profiling

- source: https://arxiv.org/abs/2504.14866
- blocker helped: packet claims need lifetime, locality, and hierarchy-aware
  accounting to be credible for accelerator readers.
- mechanism idea: split communicated object, packet lifetime, raw bytes,
  cache-line traffic, DMA traffic, and endpoint TTFT proxy.
- next experiment change: add the memory traffic ledger as a trace-card
  artifact over existing packet, text, full-log, and KV rows.
- role: systems framing / artifact design.

## C2C Cache-to-Cache Communication

- source: https://arxiv.org/abs/2510.03215
- blocker helped: the closest systems competitor is high-rate internal-state
  communication, not only text relay.
- mechanism idea: distinguish source-private endpoint packets from cache/KV
  transport by access assumption and byte floor.
- next experiment change: keep C2C/KV-style rows as reference/accounting
  comparators until native GPU cache transfer is available.
- role: baseline / competitor framing.

## Bottom Line

The memory traffic ledger should be presented as a deterministic systems
trace-card, not a production throughput benchmark. It strengthens the paper by
making the packet interface auditable: what crosses the source-target boundary,
what private state is exposed, how traffic rounds to hardware quanta, and what
is only a proxy until NVIDIA/HBM measurements are available.
