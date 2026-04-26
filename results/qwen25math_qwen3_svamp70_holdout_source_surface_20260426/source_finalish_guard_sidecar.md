# Source-Only Sidecar Router Gate

- date: `2026-04-26`
- status: `source_only_sidecar_router_fails_gate`
- reference rows: `70`
- fallback label: `target`
- preserve-on-agreement label: `none`
- source quality guard: `finalish_short_numeric`
- source quality score field: `none`
- source quality min threshold: `None`
- source quality max threshold: `None`
- source numeric coverage: `64/70`
- provenance issues: `0`

## Moduli Sweep

| Moduli | Bytes | Status | Matched | Target-Self | Clean Matched | Clean Necessary | Control Clean Union | Source-Necessary IDs | Failing Criteria |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| 2,3 | 1 | source_only_sidecar_router_fails_gate | 8 | 5 | 1 | 0 | 2 | none | min_correct, min_clean_source_necessary, max_control_clean_union |
| 2,3,5 | 1 | source_only_sidecar_router_fails_gate | 9 | 6 | 1 | 0 | 2 | none | min_correct, min_clean_source_necessary, max_control_clean_union |
| 2,3,5,7 | 1 | source_only_sidecar_router_fails_gate | 9 | 6 | 1 | 0 | 2 | none | min_correct, min_clean_source_necessary, max_control_clean_union |
| 97 | 1 | source_only_sidecar_router_fails_gate | 9 | 6 | 1 | 0 | 2 | none | min_correct, min_clean_source_necessary, max_control_clean_union |

## Interpretation

This is a source-only sidecar/router screen. The source message is formed from source-side numeric predictions only; target-side rows are used only as decoder candidate pools and controls.
