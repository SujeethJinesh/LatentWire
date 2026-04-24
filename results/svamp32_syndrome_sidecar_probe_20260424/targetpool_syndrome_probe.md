# SVAMP32 Syndrome Sidecar Probe

- date: `2026-04-24`
- status: `syndrome_sidecar_bound_clears_gate_not_method`
- reference rows: `32`
- fallback label: `target_self_repair`
- teacher numeric coverage: `32/32`
- provenance issues: `0`

## Moduli Sweep

| Moduli | Bytes | Status | Matched | Target-Only | Target-Self Matched | Clean Gold In Pool | Clean Matched | Clean Necessary | Control Clean Union | Source-Necessary IDs |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2,3 | 1 | syndrome_sidecar_bound_fails_gate | 9 | 14 | 1 | 2 | 1 | 0 | 1 | none |
| 2,3,5 | 1 | syndrome_sidecar_bound_fails_gate | 13 | 14 | 3 | 2 | 2 | 2 | 0 | `1d50b408c8f5cd2c`, `aee922049c757331` |
| 2,3,5,7 | 1 | syndrome_sidecar_bound_clears_gate_not_method | 14 | 14 | 3 | 2 | 2 | 2 | 0 | `1d50b408c8f5cd2c`, `aee922049c757331` |
| 97 | 1 | syndrome_sidecar_bound_clears_gate_not_method | 14 | 14 | 3 | 2 | 2 | 2 | 0 | `1d50b408c8f5cd2c`, `aee922049c757331` |

## Interpretation

This is a target-candidate oracle/bound probe. It uses C2C-derived numeric residues as a proxy syndrome and does not prove that a source latent can predict those residues.
