# Source-Only Sidecar Router Gate

- date: `2026-04-26`
- status: `source_only_sidecar_router_clears_gate`
- reference rows: `70`
- fallback label: `target`
- preserve-on-agreement label: `none`
- source quality guard: `finalish_short_numeric`
- source numeric coverage: `61/70`
- provenance issues: `0`

## Moduli Sweep

| Moduli | Bytes | Status | Matched | Target-Self | Clean Matched | Clean Necessary | Control Clean Union | Source-Necessary IDs | Failing Criteria |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| 2,3 | 1 | source_only_sidecar_router_fails_gate | 23 | 14 | 1 | 1 | 0 | `41cce6c6e6bb0058` | min_clean_source_necessary |
| 2,3,5 | 1 | source_only_sidecar_router_clears_gate | 25 | 15 | 2 | 2 | 0 | `41cce6c6e6bb0058`, `4d780f825bb8541c` | none |
| 2,3,5,7 | 1 | source_only_sidecar_router_clears_gate | 25 | 15 | 2 | 2 | 0 | `41cce6c6e6bb0058`, `4d780f825bb8541c` | none |
| 97 | 1 | source_only_sidecar_router_clears_gate | 25 | 15 | 2 | 2 | 0 | `41cce6c6e6bb0058`, `4d780f825bb8541c` | none |

## Interpretation

This is a source-only sidecar/router screen. The source message is formed from source-side numeric predictions only; target-side rows are used only as decoder candidate pools and controls.
