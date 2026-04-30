# Source-Private Hardware Packet Frontier

- date: `2026-04-30`
- gate: `source_private_hardware_packet_frontier`
- artifact: `results/source_private_hardware_packet_frontier_20260430/`
- status: pass as a hardware-facing systems artifact, not a production
  accelerator benchmark

## Question

Can the systems contribution be stated in terms that matter for accelerator and
hardware reviewers: memory movement, cache-line traffic, packet lifetime,
receiver contract, and explicit non-claims?

## Method

I added:

```bash
./venv_arm64/bin/python scripts/build_source_private_hardware_packet_frontier.py \
  --output-dir results/source_private_hardware_packet_frontier_20260430
```

The artifact reads the existing systems rate/assumption frontier and emits:

- raw payload bytes
- 64B cache-line rounded traffic
- 128B DMA-burst rounded traffic
- source-private / text / KV exposure assumptions
- packet lifetime and receiver access pattern
- a machine-readable packet contract (`packet_contract.json`)

## Results

Headline values:

- minimum packet raw payload: `2` bytes
- minimum packet cache-line traffic: `64` bytes
- query-aware diagnostic text: `7.0x` raw bytes but `1.0x` at 64B line granularity
- full hidden-log relay: at least `183.25x` raw bytes and `6.0x` cache-line bytes
- KV byte-floor rows: at least `10752.0x` raw bytes and `336.0x` cache-line bytes
- full-log p50 TTFT delta over packet: at least `+164.27 ms`

## Interpretation

This improves the systems story by separating semantic byte rate from hardware
traffic. A 2-byte packet is not literally a 2-byte fabric transfer on typical
systems; it still rounds to at least one cache line or DMA burst. The honest
systems claim is therefore:

> LatentWire packets minimize source-private semantic payload and avoid moving
> private text or KV/cache tensors. Even under 64B line rounding, full-log relay
> and KV/cache-style transfer remain substantially larger, while query-aware
> short text can tie a packet at line granularity but exposes private text and
> has a larger semantic payload.

This is more aligned with accelerator/hardware concerns than the earlier
byte-only table.

## Packet Contract

`packet_contract.json` records:

- byte budgets: `2`, `4`, `8`
- fields: atom/slot id, confidence/parity, optional extra atoms
- allowed receiver state: public prompt, public candidate set, public receiver
  dictionary or learned receiver weights, target-side prior/cache state
- forbidden sender material: private text relay, source KV/cache tensors, answer
  strings or candidate labels
- invalid behavior: fall back to target-only or abstain
- required controls: zero-source, shuffled-source, random same-byte,
  answer-only, answer-masked, target-derived sidecar, and derangement controls

## Non-Claims

- This is not production GPU/HBM throughput evidence.
- It does not claim superiority over native KV/cache compression.
- It does not replace the need for a less hand-built learned receiver.

## Next Systems Gate

When NVIDIA access is available, run a serving benchmark with target-only,
structured text relay, full-log relay, LatentWire packet, and a KV/cache
accounting baseline. Report TTFT, TPOT, throughput at batch sizes `1/4/16`,
prompt tokens, generated tokens, and measured HBM bytes if Nsight counters are
available.
