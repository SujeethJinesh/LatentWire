# References: Native Systems Result Ingest Gate

Web/literature check: 2026-05-02. Scope: native systems result schema,
source-exposure boundaries, and competitor rows for
`results/source_private_native_systems_result_ingest_gate_20260502/`.

## Local Artifact

- Script: `scripts/validate_source_private_native_systems_results.py`
- Result: `results/source_private_native_systems_result_ingest_gate_20260502/`
- Validator pass: `True`
- Native systems complete: `False`
- Required rows missing: `11`

## Primary Sources And Boundaries

1. vLLM / PagedAttention
   - https://arxiv.org/abs/2309.06180
   - Boundary: vLLM is a native serving substrate for TTFT, TPOT, goodput, and
     KV-memory measurements. LatentWire cannot claim native speed or memory
     wins until vLLM rows are ingested.

2. SGLang / RadixAttention
   - https://arxiv.org/abs/2312.07104
   - Boundary: SGLang is the second serving substrate for scheduler and cache
     reuse sensitivity. Mac-local byte accounting is not an SGLang result.

3. Cache-to-Cache
   - https://openreview.net/forum?id=LeatkxrBCi
   - https://arxiv.org/abs/2510.03215
   - Boundary: C2C transfers/fuses source KV cache state. It is the closest
     direct semantic-communication competitor, but it is not source-private
     under LatentWire's fixed-byte packet boundary.

4. KVComm and KVCOMM
   - https://openreview.net/forum?id=F7rUng23nw
   - https://arxiv.org/abs/2510.03346
   - https://arxiv.org/abs/2510.12872
   - Boundary: selective or online KV-cache communication baselines must be
     measured as source-state transfer rows with `source_kv_exposed=true`.

5. QJL
   - https://arxiv.org/abs/2406.03482
   - Boundary: QJL is a low-bit source-state sketch/KV-cache quantization
     baseline. It should be treated as a source-state exposure comparator, not
     a source-private packet.

6. TurboQuant
   - https://arxiv.org/abs/2504.19874
   - Boundary: TurboQuant gives the rate-distortion and low-bit vector/KV
     compression baseline. LatentWire must separate byte-floor comparisons from
     native low-bit kernel comparisons.

7. KV Packet
   - https://arxiv.org/abs/2604.13226
   - Boundary: cache packets are immutable KV/context artifacts plus adapters;
     LatentWire packets are task evidence packets and should not reuse this
     terminology without clarifying the object that crosses the boundary.

8. RelayCaching
   - https://arxiv.org/abs/2603.13289
   - Boundary: multi-agent collaboration can reuse decoding KV caches and
     recompute sparse deviations. This is a cache-reuse systems baseline, not a
     fixed-byte source-private message.

9. LMCache
   - https://arxiv.org/abs/2510.09665
   - Boundary: LMCache is relevant for cache movement/offload and serving
     integration. It strengthens the need to log source-state bytes, host-device
     traffic, and cache exposure.

10. Prompt leakage through KV-cache sharing
    - https://www.ndss-symposium.org/ndss-paper/i-know-what-you-asked-prompt-leakage-via-kv-cache-sharing-in-multi-tenant-llm-serving/
    - Boundary: source-exposure flags are not cosmetic; KV/cache sharing can
      create privacy/security leakage. This supports LatentWire's
      source-private accounting claim while limiting claims about cache baselines.

## Claim Policy

Allowed now:

- source-private boundary accounting;
- fixed packet byte ranges;
- explicit missing-native-row list;
- source-exposure distinctions between LatentWire and KV/cache competitors.

Blocked until future rows pass the ingester:

- native GPU throughput or latency wins;
- HBM/PCIe/NVLink traffic wins;
- superiority over C2C, KVComm/KVCOMM, QJL, TurboQuant, vLLM, SGLang, LMCache,
  KV Packet, or RelayCaching on their native systems surfaces.
