# Source-Private Mac Unified-Memory Transport Profile

- date: `2026-04-30`
- gate: `source_private_mac_unified_memory_transport_profile`
- artifact: `results/source_private_mac_unified_memory_transport_profile_20260430/`
- status: pass as a Mac-local boundary-traffic trace-card; not a production
  serving or native GPU throughput claim

## Question

Can the systems contribution be stated in a way that a hardware or serving
reviewer can audit: what crosses the source-target boundary, how small packets
round under realistic transfer quanta, what private state is exposed, what
prompt/KV byte floors would apply to text/log relays, and which claims remain
unmeasured?

Layman version: this asks whether the helpful hint is genuinely a tiny private
message, or whether the system is secretly moving a lot of text/cache state.

## Method

I added:

```bash
./venv_arm64/bin/python scripts/build_source_private_mac_unified_memory_transport_profile.py \
  --output-dir results/source_private_mac_unified_memory_transport_profile_20260430
```

The script joins the existing `n=160` core and held-out Qwen3-0.6B endpoint
rows with the KV/cache byte table and packet-ISA batch-packing frontier. It
emits JSON/CSV/Markdown rows for:

- raw private payload bytes
- packet record bytes with 2-byte header and 1-byte parity
- 64B cache-line and 128B DMA-burst accounting
- batch-64 packed packet traffic
- prompt-token deltas and Qwen3 prompt-KV byte floors
- p50/p95 TTFT and E2E endpoint proxy telemetry
- source-private, source-text-exposed, and source-KV-exposed flags
- exact-ID parity and reproducibility hashes

## Results

Headline values:

- pass gate: `True`
- surfaces: `2` (`core n160 label_strict`, `holdout n160 label_strict`)
- exact-ID parity: `True`
- packet payload bytes: `[2.0]`
- matched packet min delta vs target: `+0.425`
- max source-destroying-control delta vs target: `+0.000`
- query-aware text raw ratio: `7.00x`
- query-aware text 64B-line ratio: `1.00x`
- full hidden-log raw ratio: `183.25x`
- full hidden-log 64B-line ratio: `6.00x`
- full hidden-log QJL-style 1-bit prompt-KV delta / packet byte:
  `313353.6x`
- batch-64 packed packet traffic: `5.00` line bytes/request and `6.00`
  DMA bytes/request

Important row examples:

- Core matched packet: `0.675` accuracy, `2` payload bytes, `64` line bytes,
  p50 TTFT `471.6 ms`.
- Held-out matched packet: `0.688` accuracy, `2` payload bytes, `64` line
  bytes, p50 TTFT `547.2 ms`.
- Query-aware text: `14` bytes and one 64B line, but it exposes private
  evidence text.
- Full hidden log: `366.5-373.5` bytes, `384` line bytes, and
  `626707-646419` extra QJL-style prompt-KV bytes versus packet.

## Interpretation

This is now the cleanest systems-facing claim:

LatentWire packets are not magic zero-cost transfers. A single 2-byte packet
still rounds to a 64B line and 128B burst. The real systems value is that the
packet is the far-left source-private semantic payload, exposes neither private
source text nor source KV/cache, and can be packed to `5.0` line bytes/request
at batch 64. Short query-aware text can tie the packet at one line, but it is a
private-text relay and uses `7x` raw payload.

This is stronger than a byte-only table because it explicitly gives the
reviewer the caveat they would otherwise raise.

## Subagent Inputs

- Systems/hardware scout: recommended a Mac unified-memory trace-card with
  host profile, transfer quanta, KV byte floors, endpoint latency, and hard
  non-claims.
- Quantization scout: sharpened the comparison boundary against TurboQuant,
  QJL, KIVI, KVQuant, C2C, KVCOMM, and Q-KVComm. These remain native KV/cache
  or high-rate internal-state baselines, not methods LatentWire has beaten on
  their own hardware axis.
- Diffusion/JEPA scout: recommended the next method branch as a masked
  consistency receiver over learned syndrome bytes, not another prompt-only PQ
  receiver.

## Decision

Promote this as the systems contribution trace-card:

1. Source-private packet method and strict destructive controls.
2. Medium-scale packet evidence with model-mediated target decoding and
   label-blind anti-lookup defense.
3. Hardware-facing boundary accounting: raw payload, transfer quanta,
   batch-packing, prompt/KV byte floors, endpoint TTFT proxy, and exposure
   flags.

Do not use this as a production serving claim. It is a reproducible Mac-local
systems artifact that defines what must be measured on NVIDIA/server hardware:
TTFT, TPOT, goodput, and memory traffic counters under a real serving stack.

## Next Gate

Highest-value Mac-local method gate:

```text
source_private_masked_consistency_receiver_smoke_20260430
```

Start from the learned 6-byte syndrome packet, train a small one-step
byte/candidate receiver with destructive-control regularization, and run an
`n=64` smoke before widening to `n=256`. Pass requires matched >= target
`+0.15`, matched >= best destructive control `+0.15`, all controls within
target `+0.03-0.05`, and label-blind collapse.

Highest-value systems gate once hardware is available:

```text
native_serving_packet_vs_text_vs_kv_nvidia_202605xx
```

Report TTFT, TPOT, goodput, prompt/generated tokens, and hardware memory
counters for target-only, source-private packet, query-aware text,
structured/free-text relay, full-log relay, and KV/cache comparator rows.

## Tests

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_build_source_private_mac_unified_memory_transport_profile.py
```

Outcome: `2 passed in 0.15s`.
