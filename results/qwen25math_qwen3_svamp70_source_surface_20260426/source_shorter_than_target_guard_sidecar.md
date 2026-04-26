# Source-Only Sidecar Router Gate

- date: `2026-04-26`
- status: `source_only_sidecar_router_clears_gate`
- reference rows: `70`
- fallback label: `target`
- preserve-on-agreement label: `none`
- source quality guard: `shorter_than_target_numeric`
- source numeric coverage: `61/70`
- provenance issues: `0`

## Moduli Sweep

| Moduli | Bytes | Status | Matched | Target-Self | Clean Matched | Clean Necessary | Control Clean Union | Source-Necessary IDs | Failing Criteria |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| 2,3 | 1 | source_only_sidecar_router_fails_gate | 22 | 13 | 1 | 1 | 0 | `41cce6c6e6bb0058` | min_correct, min_clean_source_necessary |
| 2,3,5 | 1 | source_only_sidecar_router_fails_gate | 25 | 14 | 3 | 3 | 0 | `41cce6c6e6bb0058`, `4d780f825bb8541c`, `ce08a3a269bf0151` | min_correct, min_clean_source_necessary |
| 2,3,5,7 | 1 | source_only_sidecar_router_clears_gate | 26 | 14 | 4 | 4 | 0 | `2de1549556000830`, `41cce6c6e6bb0058`, `4d780f825bb8541c`, `ce08a3a269bf0151` | none |
| 97 | 1 | source_only_sidecar_router_clears_gate | 26 | 14 | 4 | 4 | 0 | `2de1549556000830`, `41cce6c6e6bb0058`, `4d780f825bb8541c`, `ce08a3a269bf0151` | none |

## Interpretation

This is a source-only sidecar/router screen. The source message is formed from source-side numeric predictions only; target-side rows are used only as decoder candidate pools and controls.
