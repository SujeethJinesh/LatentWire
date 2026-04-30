# Packet ISA Batch Frontier

- pass gate: `True`
- header bytes: `2`
- parity bytes: `1`
- minimum packet bytes with overhead: `5`
- max 64B-line packing efficiency: `12.80x`
- max 128B-burst packing efficiency: `21.33x`
- best min-payload line bytes/request: `5.00`

| payload | packet | batch | packed 64B lines | line bytes/request | packed 128B bursts | DMA bytes/request |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 5 | 1 | 1 | 64.00 | 1 | 128.00 |
| 2 | 5 | 4 | 1 | 16.00 | 1 | 32.00 |
| 2 | 5 | 16 | 2 | 8.00 | 1 | 8.00 |
| 2 | 5 | 64 | 5 | 5.00 | 3 | 6.00 |
| 4 | 7 | 1 | 1 | 64.00 | 1 | 128.00 |
| 4 | 7 | 4 | 1 | 16.00 | 1 | 32.00 |
| 4 | 7 | 16 | 2 | 8.00 | 1 | 8.00 |
| 4 | 7 | 64 | 7 | 7.00 | 4 | 8.00 |
| 8 | 11 | 1 | 1 | 64.00 | 1 | 128.00 |
| 8 | 11 | 4 | 1 | 16.00 | 1 | 32.00 |
| 8 | 11 | 16 | 3 | 12.00 | 2 | 16.00 |
| 8 | 11 | 64 | 11 | 11.00 | 6 | 12.00 |
| 16 | 19 | 1 | 1 | 64.00 | 1 | 128.00 |
| 16 | 19 | 4 | 2 | 32.00 | 1 | 32.00 |
| 16 | 19 | 16 | 5 | 20.00 | 3 | 24.00 |
| 16 | 19 | 64 | 19 | 19.00 | 10 | 20.00 |
| 32 | 35 | 1 | 1 | 64.00 | 1 | 128.00 |
| 32 | 35 | 4 | 3 | 48.00 | 2 | 64.00 |
| 32 | 35 | 16 | 9 | 36.00 | 5 | 40.00 |
| 32 | 35 | 64 | 35 | 35.00 | 18 | 36.00 |

## Interpretation

Single-request packet transfer is line/burst limited. Batched contiguous packet records can amortize that overhead, which is the hardware-facing reason to keep packets byte-sized even when a single request consumes one line.
