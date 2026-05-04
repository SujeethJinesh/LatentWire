# Mac Packet-Ring Transport High-Repeat Rerun

Date: 2026-05-04

## Status

- Current paper readiness: systems evidence is stronger for COLM and useful for
  ICLR framing, but it is still a Mac-local transport proxy rather than a
  native NVIDIA serving result.
- Current story: LatentWire's strongest current packet boundary is `1B` raw /
  `4B` framed source-private communication, with robust local movement
  evidence and explicit non-claims against KV/cache systems.
- Exact gap: no TTFT, TPOT, goodput, HBM traffic, or energy comparison on
  vLLM/SGLang with NVIDIA hardware yet.

## Gate

- script:
  `scripts/build_source_private_mac_packet_ring_transport_microbench.py`
- artifact:
  `results/source_private_mac_packet_ring_transport_microbench_20260504_high_repeat/`
- refreshed native ledger:
  `results/source_private_native_readiness_ledger_20260504_high_repeat/`
- refreshed evidence bundle:
  `results/source_private_iclr_evidence_bundle_20260504_high_repeat_systems/`

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/build_source_private_mac_packet_ring_transport_microbench.py \
  --output-dir results/source_private_mac_packet_ring_transport_microbench_20260504_high_repeat \
  --binary .debug/source_private_packet_ring_transport_microbench_high_repeat \
  --target-bytes 1073741824 \
  --repeats 9 \
  --min-iterations 1024 \
  --cc clang
```

## Result

Pass.

| Metric | Value |
| --- | ---: |
| repeats | `9` |
| target bytes per repeat | `1073741824` |
| packet record bytes | `4` |
| packet batch64 p50 | `0.643432 ns/request` |
| packet batch64 p95 | `0.646390 ns/request` |
| packet batch64 CV | `0.002467` |
| max packet CV | `0.023542` |
| full-log p50 ratio vs packet | `8.83x` |
| KV-floor p50 ratio vs packet | `542.33x` |
| pass gate | `true` |

The native readiness ledger now points at the high-repeat packet-ring artifact.
It still reports `native_ready=false`, `3` Mac-local measured rows, and `5`
pending native rows.

## Decision

Use this as the current systems-boundary artifact. The allowed claim is
Mac-local source-private packet byte accounting and packed-record transport
proxy evidence. The forbidden claim remains any native serving win over C2C,
KVComm, QJL, TurboQuant, KIVI/KVQuant, vLLM, or SGLang.

## Lay Explanation

This benchmark repeatedly moves tiny packet records through memory. The packet
is very small and stable in this local test, but this does not prove an
end-to-end GPU serving speedup.
