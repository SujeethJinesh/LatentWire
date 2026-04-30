# Source-Private Hardware Packet Frontier

- pass gate: `True`
- packet raw bytes min: `2.0`
- packet cache-line bytes min: `64.0`
- query-aware text raw ratio min: `7.00x`
- full-log raw ratio min: `183.25x`
- full-log cache-line ratio min: `6.00x`
- KV raw ratio min: `10752.00x`
- KV cache-line ratio min: `336.00x`
- full-log TTFT delta min: `164.27 ms`

| class | method | raw bytes | 64B lines | raw ratio | line ratio | paper use |
|---|---|---:|---:|---:|---:|---|
| endpoint_packet | LatentWire endpoint packet | 2 | 1 | 1 | 1 | headline hardware-facing packet row |
| endpoint_packet | LatentWire endpoint packet | 2 | 1 | 1 | 1 | headline hardware-facing packet row |
| endpoint_text_relay | query-aware diagnostic text | 14 | 1 | 7 | 1 | text relay traffic comparator |
| endpoint_text_relay | full hidden-log relay | 366.5 | 6 | 183.25 | 6 | text relay traffic comparator |
| endpoint_text_relay | query-aware diagnostic text | 14 | 1 | 7 | 1 | text relay traffic comparator |
| endpoint_text_relay | full hidden-log relay | 373.5 | 6 | 186.75 | 6 | text relay traffic comparator |
| semantic_anchor_medium | semantic-anchor source-private packet | 4 | 1 | 2 | 1 | method evidence traffic row |
| semantic_anchor_medium | semantic-anchor source-private packet | 8 | 1 | 4 | 1 | method evidence traffic row |
| kv_byte_floor | QJL-style 1-bit source KV byte floor | 21504 | 336 | 10752 | 336 | KV/cache movement lower-bound comparator |
| kv_byte_floor | KIVI/KVQuant-style 2-bit source KV byte floor | 43008 | 672 | 21504 | 672 | KV/cache movement lower-bound comparator |

## Contract

The packet contract is emitted as `packet_contract.json`. It records allowed receiver state, forbidden sender material, invalid-packet behavior, and required source-destroying controls.

## Non-Claims

- This is not a production accelerator throughput benchmark.
- It does not claim superiority over native KV/cache compression.
- It makes cache-line rounding explicit: tiny packets win in semantic payload bytes, while many hardware fabrics still move at least one line/burst.
