# Toy Shared Byte/Span Route Atom Remap

- Shared remap is learned from calibration examples and used before route-atom selection.
- Boundary F1 is measured against target-token boundaries on the same text; atom recovery is overlap with the oracle protected set.

| Method | Task acc | Acc delta vs token_id | Acc delta vs regroup | MSE | Boundary F1 | Remap coverage | Atom recovery | Bytes proxy | Help vs token_id | Harm vs token_id | Help vs regroup | Harm vs regroup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| token_id | 0.9167 | 0.0000 | 0.0000 | 0.0081 | 1.0000 | 0.0000 | 0.0000 | 34.9167 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| regroup_baseline | 0.9167 | 0.0000 | 0.0000 | 0.0081 | 1.0000 | 0.0000 | 0.0000 | 34.9167 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| shared_byte_span_remap_route_atoms | 0.9583 | 0.0417 | 0.0417 | 0.0028 | 1.0000 | 0.9167 | 0.6111 | 33.3333 | 0.0417 | 0.0000 | 0.0417 | 0.0000 |
| oracle_shared_byte_span_route_atoms | 0.9167 | 0.0000 | 0.0000 | 0.0005 | 0.8875 | 0.7083 | 1.0000 | 25.9167 | 0.0417 | 0.0417 | 0.0417 | 0.0417 |
