# SVAMP32 Source-Only Sidecar Router Gate

- date: `2026-04-26`
- status: `source_only_sidecar_router_clears_gate`
- reference rows: `32`
- fallback label: `target`
- preserve-on-agreement label: `t2t`
- source numeric coverage: `26/32`
- provenance issues: `0`

## Moduli Sweep

| Moduli | Bytes | Status | Matched | Target-Self | Clean Matched | Clean Necessary | Control Clean Union | Source-Necessary IDs | Failing Criteria |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| 2,3 | 1 | source_only_sidecar_router_fails_gate | 9 | 6 | 1 | 1 | 0 | `14bfbfc94f2c2e7b` | min_clean_source_necessary |
| 2,3,5 | 1 | source_only_sidecar_router_clears_gate | 10 | 6 | 2 | 2 | 0 | `14bfbfc94f2c2e7b`, `4d780f825bb8541c` | none |
| 2,3,5,7 | 1 | source_only_sidecar_router_clears_gate | 11 | 6 | 3 | 3 | 0 | `14bfbfc94f2c2e7b`, `2de1549556000830`, `4d780f825bb8541c` | none |
| 97 | 1 | source_only_sidecar_router_clears_gate | 11 | 6 | 3 | 3 | 0 | `14bfbfc94f2c2e7b`, `2de1549556000830`, `4d780f825bb8541c` | none |

## Interpretation

This is a source-only sidecar/router screen. The source message is formed from source-side numeric predictions only; target-side rows are used only as decoder candidate pools and controls.
