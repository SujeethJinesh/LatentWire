# SinkKV Deterministic Probe

- seed: `20260506`
- decision: `SYNTHETIC_PASS_REAL_DUMPS_NEXT`
- sink-vs-uniform recovery: `0.299`
- recent-vs-uniform recovery: `-0.160`

This is a synthetic-only validation. It is not GPU speed evidence, not benchmark accuracy, and it does not skip QK_sink.

| row | bits/element | output rel-L2 | softmax L1 | sink mass |
|---|---:|---:|---:|---:|
| full_precision_kv | 16.000 | 0.000000 | 0.000000 | 0.087174 |
| uniform_mxfp4_kv | 4.000 | 0.097249 | 0.000147 | 0.087221 |
| sink_protected_budget_matched_kv | 4.000 | 0.068215 | 0.000207 | 0.087242 |
| recent_protected_budget_matched_kv | 4.000 | 0.112774 | 0.000217 | 0.087222 |
