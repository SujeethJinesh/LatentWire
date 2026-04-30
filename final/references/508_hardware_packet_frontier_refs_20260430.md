# Hardware Packet Frontier References

- date: `2026-04-30`
- purpose: primary-source grounding for the hardware-facing LatentWire packet
  frontier and future accelerator/system experiments.

## Tambe Lab Research Themes

- source: https://tambelab.stanford.edu/research
- blocker helped: the systems contribution needs to speak in terms of
  algorithm-hardware co-design, data movement, and memory hierarchy, not only
  model accuracy.
- mechanism idea: describe LatentWire packets as an explicit communication
  interface with a contract, lifetime, byte traffic, and memory movement.
- next experiment change: add `packet_contract.json` plus cache-line/DMA traffic
  accounting.
- role: paper framing / systems motivation.

## GainSight / Heterogeneous On-Chip Memory Profiling

- source: https://arxiv.org/abs/2504.14866
- blocker helped: byte-only claims can be misleading without memory lifetime and
  locality accounting.
- mechanism idea: report source-private packets, text relay, and KV/cache rows
  by lifetime, access pattern, and rounded memory transfer granularity.
- next experiment change: add 64B cache-line and 128B DMA-burst estimates.
- role: systems framing / ablation design.

## BlockDialect / Mixed-Format LLM Quantization

- source: https://arxiv.org/abs/2501.01144
- blocker helped: reviewers may ask whether fixed byte packets are hardware
  arbitrary rather than accelerator-friendly.
- mechanism idea: treat packet fields as typed low-bit lanes: slot id,
  confidence/parity, optional residual evidence.
- next experiment change: future typed-packet sweep should compare fixed
  `uint8` fields, parity, and confidence-coded packets.
- role: quantization inspiration / future ablation.

## EdgeBERT

- source: https://arxiv.org/abs/2011.14203
- blocker helped: local CPU TTFT is not enough for accelerator systems claims;
  adaptive compute and early exit are the right systems comparison language.
- mechanism idea: split source packet generation, target decode, valid rate,
  and latency/energy proxy metrics.
- next experiment change: future GPU/NVIDIA run should report TTFT, TPOT,
  throughput, batch size, and measured memory traffic.
- role: systems framing.

## AdaptivFloat

- source: https://arxiv.org/abs/1909.13271
- blocker helped: tiny numeric packet fields need a principled low-precision
  representation story.
- mechanism idea: dynamic range, clipping, and field-specific precision can be
  part of packet design rather than incidental serialization.
- next experiment change: add packet-format sweeps only after the receiver is
  stronger.
- role: quantization inspiration.

## CAMEL / AI Model and eDRAM Co-Design

- source: https://arxiv.org/abs/2305.03148
- blocker helped: supports the framing that full transient state movement is a
  hardware cost, not only a software inconvenience.
- mechanism idea: communicate compact transient evidence instead of retaining
  or transmitting full source activations/KV.
- next experiment change: keep full-log and KV/cache movement rows in the
  systems frontier.
- role: systems framing.

## 3LA / Software-Hardware Interface Validation

- source: https://arxiv.org/abs/2203.00218
- blocker helped: packet methods need a precise interface contract or reviewers
  will view them as informal prompt conventions.
- mechanism idea: specify the packet ISA: fields, receiver state, forbidden
  sender material, invalid behavior, and required controls.
- next experiment change: emit machine-readable `packet_contract.json`.
- role: theory/framing.

## Bottom Line

For a systems-heavy ICLR submission, the strongest safe contribution is not a
new accelerator claim. It is an explicit, low-rate source-private communication
interface with memory-traffic accounting and a packet contract that future
hardware experiments can implement.
