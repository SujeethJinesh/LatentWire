# Source-Private Rate Frontier

- pass gate: `True`
- packet oracle bytes max: `2.0`
- JSON/free-text oracle bytes min: `21.0` / `17.0`
- query-aware diagnostic-span oracle bytes min: `14.0`
- packet vs JSON oracle compression min: `10.5x`
- packet vs query-aware oracle compression min: `7.0x`
- packet vs full hidden-log compression min: `183.2x`
- matched-byte text at packet accuracy max: `0.250`

## Per-Surface Frontier

| Surface | Target | Packet oracle bytes | JSON oracle bytes | Free-text oracle bytes | Query-aware oracle bytes | Full log bytes | Packet vs JSON | Packet vs query-aware | Packet vs full log | Matched-byte text at packet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core seed29 | 0.250 | 2.0 | 21.0 | 17.0 | 14.0 | 366.4 | 10.5x | 7.0x | 183.2x | 0.250 |
| holdout seed30 | 0.250 | 2.0 | 21.0 | 17.0 | 14.0 | 373.5 | 10.5x | 7.0x | 186.7x | 0.250 |

## Rate Rows

| Surface | Budget | Interface | Kind | Accuracy | Bytes | Tokens | p50 ms |
|---|---:|---|---|---:|---:|---:|---:|
| core seed29 | 2 | target-only | no source | 0.250 | 0.0 | 0.0 | 0.001 |
| core seed29 | 2 | diagnostic packet | method | 1.000 | 2.0 | 1.0 | 0.002 |
| core seed29 | 2 | hidden-log truncation | matched-byte text | 0.250 | 2.0 | 1.0 | 0.002 |
| core seed29 | 2 | JSON relay | matched-byte text | 0.250 | 2.0 | 1.0 | 0.004 |
| core seed29 | 2 | free-text relay | matched-byte text | 0.250 | 2.0 | 1.0 | 0.002 |
| core seed29 | 2 | full hidden-log relay | oracle text relay | 1.000 | 366.4 | 33.9 | 0.006 |
| core seed29 | 2 | full diagnostic text | oracle diagnostic text | 1.000 | 14.0 | 1.0 | 0.002 |
| core seed29 | 4 | target-only | no source | 0.250 | 0.0 | 0.0 | 0.001 |
| core seed29 | 4 | diagnostic packet | method | 1.000 | 2.0 | 1.0 | 0.002 |
| core seed29 | 4 | hidden-log truncation | matched-byte text | 0.250 | 4.0 | 1.0 | 0.002 |
| core seed29 | 4 | JSON relay | matched-byte text | 0.250 | 4.0 | 1.0 | 0.004 |
| core seed29 | 4 | free-text relay | matched-byte text | 0.250 | 4.0 | 1.0 | 0.002 |
| core seed29 | 4 | full hidden-log relay | oracle text relay | 1.000 | 366.4 | 33.9 | 0.006 |
| core seed29 | 4 | full diagnostic text | oracle diagnostic text | 1.000 | 14.0 | 1.0 | 0.002 |
| core seed29 | 8 | target-only | no source | 0.250 | 0.0 | 0.0 | 0.001 |
| core seed29 | 8 | diagnostic packet | method | 1.000 | 2.0 | 1.0 | 0.002 |
| core seed29 | 8 | hidden-log truncation | matched-byte text | 0.250 | 8.0 | 2.0 | 0.002 |
| core seed29 | 8 | JSON relay | matched-byte text | 0.250 | 8.0 | 1.0 | 0.004 |
| core seed29 | 8 | free-text relay | matched-byte text | 0.250 | 8.0 | 2.0 | 0.002 |
| core seed29 | 8 | full hidden-log relay | oracle text relay | 1.000 | 366.4 | 33.9 | 0.006 |
| core seed29 | 8 | full diagnostic text | oracle diagnostic text | 1.000 | 14.0 | 1.0 | 0.002 |
| core seed29 | 16 | target-only | no source | 0.250 | 0.0 | 0.0 | 0.001 |
| core seed29 | 16 | diagnostic packet | method | 1.000 | 2.0 | 1.0 | 0.002 |
| core seed29 | 16 | hidden-log truncation | matched-byte text | 0.250 | 16.0 | 3.0 | 0.003 |
| core seed29 | 16 | JSON relay | matched-byte text | 0.250 | 16.0 | 1.0 | 0.004 |
| core seed29 | 16 | free-text relay | matched-byte text | 0.250 | 16.0 | 4.0 | 0.003 |
| core seed29 | 16 | full hidden-log relay | oracle text relay | 1.000 | 366.4 | 33.9 | 0.006 |
| core seed29 | 16 | full diagnostic text | oracle diagnostic text | 1.000 | 14.0 | 1.0 | 0.002 |
| core seed29 | 32 | target-only | no source | 0.250 | 0.0 | 0.0 | 0.001 |
| core seed29 | 32 | diagnostic packet | method | 1.000 | 2.0 | 1.0 | 0.002 |
| core seed29 | 32 | hidden-log truncation | matched-byte text | 0.250 | 32.0 | 4.0 | 0.003 |
| core seed29 | 32 | JSON relay | matched-byte text | 1.000 | 21.0 | 2.0 | 0.004 |
| core seed29 | 32 | free-text relay | matched-byte text | 1.000 | 17.0 | 4.0 | 0.003 |
| core seed29 | 32 | full hidden-log relay | oracle text relay | 1.000 | 366.4 | 33.9 | 0.006 |
| core seed29 | 32 | full diagnostic text | oracle diagnostic text | 1.000 | 14.0 | 1.0 | 0.002 |
| holdout seed30 | 2 | target-only | no source | 0.250 | 0.0 | 0.0 | 0.001 |
| holdout seed30 | 2 | diagnostic packet | method | 1.000 | 2.0 | 1.0 | 0.002 |
| holdout seed30 | 2 | hidden-log truncation | matched-byte text | 0.250 | 2.0 | 1.0 | 0.002 |
| holdout seed30 | 2 | JSON relay | matched-byte text | 0.250 | 2.0 | 1.0 | 0.004 |
| holdout seed30 | 2 | free-text relay | matched-byte text | 0.250 | 2.0 | 1.0 | 0.002 |
| holdout seed30 | 2 | full hidden-log relay | oracle text relay | 1.000 | 373.5 | 34.8 | 0.006 |
| holdout seed30 | 2 | full diagnostic text | oracle diagnostic text | 1.000 | 14.0 | 1.0 | 0.002 |
| holdout seed30 | 4 | target-only | no source | 0.250 | 0.0 | 0.0 | 0.001 |
| holdout seed30 | 4 | diagnostic packet | method | 1.000 | 2.0 | 1.0 | 0.002 |
| holdout seed30 | 4 | hidden-log truncation | matched-byte text | 0.250 | 4.0 | 1.0 | 0.002 |
| holdout seed30 | 4 | JSON relay | matched-byte text | 0.250 | 4.0 | 1.0 | 0.004 |
| holdout seed30 | 4 | free-text relay | matched-byte text | 0.250 | 4.0 | 1.0 | 0.002 |
| holdout seed30 | 4 | full hidden-log relay | oracle text relay | 1.000 | 373.5 | 34.8 | 0.006 |
| holdout seed30 | 4 | full diagnostic text | oracle diagnostic text | 1.000 | 14.0 | 1.0 | 0.002 |
| holdout seed30 | 8 | target-only | no source | 0.250 | 0.0 | 0.0 | 0.001 |
| holdout seed30 | 8 | diagnostic packet | method | 1.000 | 2.0 | 1.0 | 0.002 |
| holdout seed30 | 8 | hidden-log truncation | matched-byte text | 0.250 | 8.0 | 2.0 | 0.002 |
| holdout seed30 | 8 | JSON relay | matched-byte text | 0.250 | 8.0 | 1.0 | 0.004 |
| holdout seed30 | 8 | free-text relay | matched-byte text | 0.250 | 8.0 | 2.0 | 0.002 |
| holdout seed30 | 8 | full hidden-log relay | oracle text relay | 1.000 | 373.5 | 34.8 | 0.006 |
| holdout seed30 | 8 | full diagnostic text | oracle diagnostic text | 1.000 | 14.0 | 1.0 | 0.002 |
| holdout seed30 | 16 | target-only | no source | 0.250 | 0.0 | 0.0 | 0.001 |
| holdout seed30 | 16 | diagnostic packet | method | 1.000 | 2.0 | 1.0 | 0.002 |
| holdout seed30 | 16 | hidden-log truncation | matched-byte text | 0.250 | 16.0 | 3.0 | 0.003 |
| holdout seed30 | 16 | JSON relay | matched-byte text | 0.250 | 16.0 | 1.0 | 0.004 |
| holdout seed30 | 16 | free-text relay | matched-byte text | 0.250 | 16.0 | 4.0 | 0.003 |
| holdout seed30 | 16 | full hidden-log relay | oracle text relay | 1.000 | 373.5 | 34.8 | 0.006 |
| holdout seed30 | 16 | full diagnostic text | oracle diagnostic text | 1.000 | 14.0 | 1.0 | 0.002 |
| holdout seed30 | 32 | target-only | no source | 0.250 | 0.0 | 0.0 | 0.001 |
| holdout seed30 | 32 | diagnostic packet | method | 1.000 | 2.0 | 1.0 | 0.002 |
| holdout seed30 | 32 | hidden-log truncation | matched-byte text | 0.250 | 32.0 | 4.0 | 0.003 |
| holdout seed30 | 32 | JSON relay | matched-byte text | 1.000 | 21.0 | 2.0 | 0.004 |
| holdout seed30 | 32 | free-text relay | matched-byte text | 1.000 | 17.0 | 4.0 | 0.003 |
| holdout seed30 | 32 | full hidden-log relay | oracle text relay | 1.000 | 373.5 | 34.8 | 0.006 |
| holdout seed30 | 32 | full diagnostic text | oracle diagnostic text | 1.000 | 14.0 | 1.0 | 0.002 |
| core seed29 | 2 | query-aware diagnostic span | query-aware compressed text | 0.250 | 2.0 | 1.0 | 0.001 |
| core seed29 | 2 | query-aware masked diagnostic span | query-aware text control | 0.250 | 2.0 | 1.0 | 0.001 |
| core seed29 | 4 | query-aware diagnostic span | query-aware compressed text | 0.250 | 4.0 | 1.0 | 0.001 |
| core seed29 | 4 | query-aware masked diagnostic span | query-aware text control | 0.250 | 4.0 | 1.0 | 0.001 |
| core seed29 | 8 | query-aware diagnostic span | query-aware compressed text | 0.250 | 8.0 | 1.0 | 0.001 |
| core seed29 | 8 | query-aware masked diagnostic span | query-aware text control | 0.250 | 8.0 | 1.0 | 0.001 |
| core seed29 | 16 | query-aware diagnostic span | query-aware compressed text | 1.000 | 14.0 | 1.0 | 0.001 |
| core seed29 | 16 | query-aware masked diagnostic span | query-aware text control | 0.250 | 14.0 | 1.0 | 0.001 |
| core seed29 | 32 | query-aware diagnostic span | query-aware compressed text | 1.000 | 14.0 | 1.0 | 0.001 |
| core seed29 | 32 | query-aware masked diagnostic span | query-aware text control | 0.250 | 14.0 | 1.0 | 0.001 |
| holdout seed30 | 2 | query-aware diagnostic span | query-aware compressed text | 0.250 | 2.0 | 1.0 | 0.001 |
| holdout seed30 | 2 | query-aware masked diagnostic span | query-aware text control | 0.250 | 2.0 | 1.0 | 0.001 |
| holdout seed30 | 4 | query-aware diagnostic span | query-aware compressed text | 0.250 | 4.0 | 1.0 | 0.001 |
| holdout seed30 | 4 | query-aware masked diagnostic span | query-aware text control | 0.250 | 4.0 | 1.0 | 0.001 |
| holdout seed30 | 8 | query-aware diagnostic span | query-aware compressed text | 0.250 | 8.0 | 1.0 | 0.001 |
| holdout seed30 | 8 | query-aware masked diagnostic span | query-aware text control | 0.250 | 8.0 | 1.0 | 0.001 |
| holdout seed30 | 16 | query-aware diagnostic span | query-aware compressed text | 1.000 | 14.0 | 1.0 | 0.001 |
| holdout seed30 | 16 | query-aware masked diagnostic span | query-aware text control | 0.250 | 14.0 | 1.0 | 0.001 |
| holdout seed30 | 32 | query-aware diagnostic span | query-aware compressed text | 1.000 | 14.0 | 1.0 | 0.001 |
| holdout seed30 | 32 | query-aware masked diagnostic span | query-aware text control | 0.250 | 14.0 | 1.0 | 0.001 |

## Caveat

Latency is local Python single-request timing; this artifact proves rate frontier, not endpoint TTFT.
