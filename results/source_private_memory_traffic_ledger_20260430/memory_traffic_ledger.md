# Source-Private Memory Traffic Ledger

- pass gate: `True`
- packet raw bytes min: `2.00`
- packet single-request cache-line bytes min: `64.00`
- packet batch-64 line bytes/request min: `5.00`
- query-aware text raw ratio: `7.00x`
- query-aware text cache-line ratio: `1.00x`
- full-log raw ratio: `183.25x`
- full-log cache-line ratio: `6.00x`
- KV raw ratio: `10752.00x`
- KV cache-line ratio: `336.00x`

| class | method | raw B | line B | batch line B/req | DMA B | TTFT delta ms | exposure | conclusion |
|---|---|---:|---:|---:|---:|---:|---|---|
| endpoint_packet | LatentWire endpoint packet | 2.00 | 64.00 | 5.00 | 128.00 | 0.00 | source-private | source-private packet stays below one cache line when batched and avoids text/KV movement |
| endpoint_packet | LatentWire endpoint packet | 2.00 | 64.00 | 5.00 | 128.00 | 0.00 | source-private | source-private packet stays below one cache line when batched and avoids text/KV movement |
| endpoint_text_relay | query-aware diagnostic text | 14.00 | 64.00 |  | 128.00 |  | text | query-aware text ties one cache line but exposes private text and uses 7x raw bytes |
| endpoint_text_relay | full hidden-log relay | 366.50 | 384.00 |  | 384.00 | 164.27 | text | full private text relay costs multiple cache lines and adds endpoint TTFT |
| endpoint_text_relay | query-aware diagnostic text | 14.00 | 64.00 |  | 128.00 |  | text | query-aware text ties one cache line but exposes private text and uses 7x raw bytes |
| endpoint_text_relay | full hidden-log relay | 373.50 | 384.00 |  | 384.00 | 183.54 | text | full private text relay costs multiple cache lines and adds endpoint TTFT |
| semantic_anchor_medium | semantic-anchor source-private packet | 4.00 | 64.00 | 7.00 | 128.00 |  | source-private | source-private packet stays below one cache line when batched and avoids text/KV movement |
| semantic_anchor_medium | semantic-anchor source-private packet | 8.00 | 64.00 | 11.00 | 128.00 |  | source-private | source-private packet stays below one cache line when batched and avoids text/KV movement |
| kv_byte_floor | QJL-style 1-bit source KV byte floor | 21504.00 | 21504.00 |  | 21504.00 |  | KV | KV/cache transport is a lower-bound accounting row, not a native benchmark |
| kv_byte_floor | KIVI/KVQuant-style 2-bit source KV byte floor | 43008.00 | 43008.00 |  | 43008.00 |  | KV | KV/cache transport is a lower-bound accounting row, not a native benchmark |

## Interpretation

The packet advantage is strongest as semantic payload and private-state movement: packets carry 2-8 bytes and expose neither private text nor source KV/cache tensors. A single request still rounds to at least one 64B cache line, so this artifact explicitly marks query-aware text as a line-granularity tie while preserving the raw-byte and privacy distinction. Batched contiguous packet records amortize the line cost to 5.0 bytes/request for the 2-byte payload plus header/parity.
