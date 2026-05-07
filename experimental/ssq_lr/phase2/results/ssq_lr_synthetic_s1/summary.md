# SSQ-LR Synthetic S1 Real-Schema Rehearsal

Decision: `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_SSQ_LR_S1`.

This packet is intentionally a schema rehearsal and non-promotable.
It validates the real S1 row schema, provenance fields, and
recomputed evaluator summary using synthetic CPU tensors.

Fixture-only evaluator fields, not a promotion decision:
- `gate_status`: `PASS_REAL_S1_HETEROGENEITY` (raw evaluator readout only; decision remains `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_SSQ_LR_S1`)
- `prompt_count`: `12`
- `ssm_layer_count`: `6`
- `selected_s1_ratio`: `3.889`
- `selected_s1_ci_low`: `3.889`
