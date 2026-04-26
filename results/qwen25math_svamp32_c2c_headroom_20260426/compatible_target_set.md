# SVAMP32 Innovation Target Set

- date: `2026-04-26`
- status: `residual_headroom_available`
- target: `8/32`
- C2C teacher: `15/32`
- target_self_repair: `0/32`
- C2C-only IDs: `9`
- clean residual C2C-only targets: `6`
- oracle target_self_repair plus C2C teacher: `9/32`
- required clean residual wins if preserving target_self_repair: `0`

## Interpretation

A target-self-preserving connector can clear the current paper gate by adding the required number of clean residual C2C-only wins.

## ID Sets

- target_self_repair C2C-only: none
- source/source-control explained C2C-only: `14bfbfc94f2c2e7b`, `2de1549556000830`, `4d780f825bb8541c`
- clean residual targets: `1d50b408c8f5cd2c`, `3e8a5691f5443495`, `47464cc0b064f172`, `575d7e83d84c1e67`, `6e9745b37ab6fc45`, `de1bf4d142544e5b`

## Teacher-Only Provenance

| Example ID | Labels | Source correct | Control correct | Candidate correct |
|---|---|---|---|---|
| 14bfbfc94f2c2e7b | `source_or_text` | `source` | none | none |
| 1d50b408c8f5cd2c | `clean_c2c_residual_target` | none | none | none |
| 2de1549556000830 | `source_or_text` | `source` | none | none |
| 3e8a5691f5443495 | `clean_c2c_residual_target` | none | none | none |
| 47464cc0b064f172 | `clean_c2c_residual_target` | none | none | none |
| 4d780f825bb8541c | `source_or_text` | `source` | none | none |
| 575d7e83d84c1e67 | `clean_c2c_residual_target` | none | none | none |
| 6e9745b37ab6fc45 | `clean_c2c_residual_target` | none | none | none |
| de1bf4d142544e5b | `clean_c2c_residual_target` | none | none | none |
