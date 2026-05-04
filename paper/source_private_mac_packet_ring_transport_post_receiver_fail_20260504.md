# 2026-05-04 Post-Receiver-Failure Mac Packet-Ring Systems Gate

## Status

This is a supporting systems artifact, not a positive learned-method result.
It strengthens the paper's source-private byte/transport story after the latest
target-native receiver branches failed source-specific controls.

## Gate

Rerun the Mac packet-ring transport microbenchmark with high-repeat settings:

```bash
./venv_arm64/bin/python scripts/build_source_private_mac_packet_ring_transport_microbench.py \
  --output-dir results/source_private_mac_packet_ring_transport_microbench_20260504_post_receiver_fail \
  --binary .debug/source_private_packet_ring_transport_microbench_post_receiver_fail \
  --target-bytes 1073741824 \
  --repeats 9 \
  --min-iterations 1024 \
  --cc clang
```

Then regenerate the native-readiness ledger against the same artifact:

```bash
./venv_arm64/bin/python scripts/build_source_private_native_readiness_ledger.py \
  --packet-ring results/source_private_mac_packet_ring_transport_microbench_20260504_post_receiver_fail/packet_ring_transport_microbench.json \
  --output-dir results/source_private_native_readiness_ledger_20260504_post_receiver_fail
```

## Result

The packet-ring gate passes as a local Mac transport proxy:

- packet profile: `1B` payload / `4B` framed record;
- batch64 packet p50: `0.642475 ns/request`;
- batch64 packet p95: `0.646144 ns/request`;
- batch64 line/DMA bytes: `4.0/4.0 B/request`;
- packet batch64 CV: `0.002493`;
- max packet CV: `0.024077`;
- PQ packet profile: `4B` payload / `7B` framed record;
- PQ batch64 p95: `0.680982 ns/request`;
- PQ line/DMA bytes: `7.0/8.0 B/request`;
- full private hidden-log p50 ratio vs packet: `8.896008x`;
- QJL 1-bit KV-floor p50 ratio vs packet: `593.085303x`.

The native-readiness ledger intentionally fails native readiness:

- local measured rows: `3`;
- pending native rows: `5`;
- blocker: NVIDIA/vLLM/SGLang run is still required for systems claims.

Artifacts:

- `results/source_private_mac_packet_ring_transport_microbench_20260504_post_receiver_fail/`;
- `results/source_private_native_readiness_ledger_20260504_post_receiver_fail/`.

## Interpretation

Safe claim: LatentWire now has measured Mac-local evidence that fixed-byte
source-private packet records can be packed, copied, and verified with stable
sub-microsecond per-request overhead on a batch64 transport microbenchmark.
This supports a systems-side boundary row for packet framing, byte exposure,
and local movement cost.

Forbidden claim: this does not prove native GPU serving speedup, HBM traffic
reduction, peak-memory reduction, TTFT/TPOT/goodput improvement, vLLM/SGLang
integration, or superiority over C2C, KVComm/KVCOMM, QJL, TurboQuant, KIVI, or
other native KV-cache methods.

## Decision

Promote the post-receiver-failure packet-ring run as the current canonical
Mac-local systems artifact. Keep native systems as an explicit blocker for
ICLR. The next positive-method branch should follow the subagent recommendation:
a receiver-calibrated sparse/common-basis top1/top2 ambiguity code that sends
conditional evidence for a concrete source-target disagreement, not a generic
global atom.

Lay explanation: this experiment repeatedly moves the tiny LatentWire packet
through local memory and checks that it arrives correctly. It shows that the
packet is very small and locally cheap to move. It does not show that a GPU
server will be faster until we run the native serving benchmark.
