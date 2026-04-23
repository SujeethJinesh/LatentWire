# SVAMP32 Innovation Target Set

- date: `2026-04-23`
- status: `residual_headroom_available`
- target: `8/32`
- C2C teacher: `16/32`
- target_self_repair: `14/32`
- C2C-only IDs: `10`
- clean residual C2C-only targets: `6`
- oracle target_self_repair plus C2C teacher: `21/32`
- required clean residual wins if preserving target_self_repair: `2`

## Interpretation

A target-self-preserving connector can clear the current paper gate by adding the required number of clean residual C2C-only wins.

## ID Sets

- target_self_repair C2C-only: `4c84ebf42812703b`, `4d780f825bb8541c`, `de1bf4d142544e5b`
- source/source-control explained C2C-only: `575d7e83d84c1e67`
- clean residual targets: `13cb77b698eeadb5`, `1d50b408c8f5cd2c`, `2de1549556000830`, `6e9745b37ab6fc45`, `aee922049c757331`, `e3ab8666238a289e`

## Teacher-Only Provenance

| Example ID | Labels | Source correct | Control correct | Candidate correct |
|---|---|---|---|---|
| 13cb77b698eeadb5 | `clean_c2c_residual_target` | none | none | none |
| 1d50b408c8f5cd2c | `clean_c2c_residual_target` | none | none | none |
| 2de1549556000830 | `clean_c2c_residual_target` | none | none | none |
| 4c84ebf42812703b | `target_self_repair` | none | `target_self_repair` | none |
| 4d780f825bb8541c | `target_self_repair` | none | `selected_route_no_repair`, `target_self_repair` | none |
| 575d7e83d84c1e67 | `source_or_text`, `source_control`, `current_candidate` | `source` | `shuffled_source`, `zero_source` | `query_pool_matched` |
| 6e9745b37ab6fc45 | `clean_c2c_residual_target` | none | none | none |
| aee922049c757331 | `clean_c2c_residual_target` | none | none | none |
| de1bf4d142544e5b | `target_self_repair` | none | `target_self_repair` | none |
| e3ab8666238a289e | `clean_c2c_residual_target` | none | none | none |
