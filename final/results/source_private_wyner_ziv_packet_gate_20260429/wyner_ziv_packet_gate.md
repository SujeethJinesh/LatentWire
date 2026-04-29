# Source-Private Wyner-Ziv Packet Gate

- pass gate: `True`
- rows: `9`
- pass rows: `9`
- remap seeds: `[101, 103, 107]`
- budgets: `[2, 4, 6]`
- minimum passing scalar accuracy: `0.418`
- minimum passing scalar-control margin: `0.154`
- minimum packet-vs-query-aware text compression: `2.3x`

## Rows

| Remap | Budget | N | Scalar WZ | Target | Best scalar control | Raw sign | QJL | Canonical RASP | Query-aware text@budget | Packet/query-aware oracle | Scalar pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 2 | 512 | 0.418 | 0.250 | 0.264 | 0.301 | 0.396 | 0.350 | 0.250 | 7.0x | `True` |
| 101 | 4 | 512 | 0.432 | 0.250 | 0.264 | 0.326 | 0.461 | 0.494 | 0.250 | 3.5x | `True` |
| 101 | 6 | 512 | 0.463 | 0.250 | 0.264 | 0.332 | 0.447 | 0.494 | 0.250 | 2.3x | `True` |
| 103 | 2 | 512 | 0.436 | 0.250 | 0.250 | 0.303 | 0.439 | 0.363 | 0.250 | 7.0x | `True` |
| 103 | 4 | 512 | 0.475 | 0.250 | 0.266 | 0.328 | 0.461 | 0.520 | 0.250 | 3.5x | `True` |
| 103 | 6 | 512 | 0.508 | 0.250 | 0.266 | 0.316 | 0.484 | 0.520 | 0.250 | 2.3x | `True` |
| 107 | 2 | 512 | 0.418 | 0.250 | 0.246 | 0.309 | 0.393 | 0.350 | 0.250 | 7.0x | `True` |
| 107 | 4 | 512 | 0.445 | 0.250 | 0.246 | 0.326 | 0.453 | 0.506 | 0.250 | 3.5x | `True` |
| 107 | 6 | 512 | 0.492 | 0.250 | 0.232 | 0.330 | 0.457 | 0.506 | 0.250 | 2.3x | `True` |

## Interpretation

This is a learned source-private syndrome gate: the encoder maps private source evidence into a compact vector packet, while the decoder uses public candidate side information. It is less hand-coded than the deterministic diagnostic packet, but still scoped to same-family/all-family remapped slot codebooks.
