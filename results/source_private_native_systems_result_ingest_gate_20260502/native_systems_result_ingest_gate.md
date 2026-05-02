# Native Systems Result Ingest Gate

- validator pass: `True`
- native systems complete: `False`
- paper native win allowed: `False`
- measurement rows: `0`
- required baseline rows: `11`
- missing required rows: `11`
- invalid measurement rows: `0`

## Missing Required Rows

- `c2c_cache_to_cache`
- `kvcomm_online_cross_context`
- `kvcomm_selective_kv`
- `latentwire_packet_cached_source`
- `latentwire_packet_end_to_end_source_scoring`
- `qjl_1bit_source_state`
- `same_byte_visible_text`
- `source_label_copy_control`
- `target_only_sglang`
- `target_only_vllm`
- `turboquant_lowbit_source_state`

## Claim Boundary Matrix

The companion `native_systems_claim_boundary_matrix.csv` states which claims are allowed for each required row. In the current no-measurement state it allows Mac-local byte/exposure accounting and forbids throughput, HBM, latency, and native-serving win claims.

## Decision

Native systems claims remain blocked until every required baseline row is ingested with all quality, latency, memory, traffic, payload-byte, and source-exposure fields. The current run validates the schema and correctly refuses native-systems-complete because required native measurement rows are missing.

## Lay Explanation

This checker is a checklist with teeth. It will not let the paper say we have a real GPU systems win until every required method has the same accuracy, latency, memory, traffic, byte, and privacy fields filled in.
