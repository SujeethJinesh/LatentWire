# Source-Only Sidecar Router Gate

- date: `2026-04-26`
- status: `source_only_sidecar_router_clears_gate`
- reference rows: `70`
- fallback label: `target`
- preserve-on-agreement label: `t2t`
- source numeric coverage: `61/70`
- provenance issues: `0`

## Moduli Sweep

| Moduli | Bytes | Status | Matched | Target-Self | Clean Matched | Clean Necessary | Control Clean Union | Source-Necessary IDs | Failing Criteria |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| 2,3 | 1 | source_only_sidecar_router_fails_gate | 22 | 15 | 2 | 2 | 0 | `14bfbfc94f2c2e7b`, `bd9d8da923981d69` | min_correct, min_clean_source_necessary |
| 2,3,5 | 1 | source_only_sidecar_router_clears_gate | 24 | 16 | 3 | 3 | 0 | `14bfbfc94f2c2e7b`, `4d780f825bb8541c`, `bd9d8da923981d69` | none |
| 2,3,5,7 | 1 | source_only_sidecar_router_clears_gate | 25 | 16 | 4 | 4 | 0 | `14bfbfc94f2c2e7b`, `2de1549556000830`, `4d780f825bb8541c`, `bd9d8da923981d69` | none |
| 97 | 1 | source_only_sidecar_router_clears_gate | 25 | 16 | 4 | 4 | 0 | `14bfbfc94f2c2e7b`, `2de1549556000830`, `4d780f825bb8541c`, `bd9d8da923981d69` | none |

## Interpretation

This is a source-only sidecar/router screen. The source message is formed from source-side numeric predictions only; target-side rows are used only as decoder candidate pools and controls.
