# SVAMP32 Source-Only Sidecar Router Gate

- date: `2026-04-26`
- status: `source_only_sidecar_router_fails_gate`
- reference rows: `32`
- fallback label: `target_self_repair`
- source numeric coverage: `32/32`
- provenance issues: `0`

## Moduli Sweep

| Moduli | Bytes | Status | Matched | Target-Self | Clean Matched | Clean Necessary | Control Clean Union | Source-Necessary IDs | Failing Criteria |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| 2,3,5,7 | 1 | source_only_sidecar_router_fails_gate | 4 | 0 | 0 | 0 | 0 | none | min_correct, min_target_self, min_clean_source_necessary |
| 97 | 1 | source_only_sidecar_router_fails_gate | 4 | 0 | 0 | 0 | 0 | none | min_correct, min_target_self, min_clean_source_necessary |

## Interpretation

This is a source-only sidecar/router screen. The source message is formed from source-side numeric predictions only; target-side rows are used only as decoder candidate pools and controls.
