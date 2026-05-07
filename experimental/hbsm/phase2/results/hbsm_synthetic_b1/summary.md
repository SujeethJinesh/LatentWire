# HBSM Synthetic B1 Real-Schema Rehearsal

Decision: `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HBSM_B1`.

This packet is intentionally a schema rehearsal and non-promotable.
It validates the real B1 row schema, controls, prompt-to-layer
aggregation, and recomputed evaluator summary using synthetic rows.

Fixture-only evaluator fields, not a promotion decision:
- `gate_status`: `PASS_REAL_B1_SENSITIVITY_HETEROGENEITY` (raw evaluator readout only; decision remains `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HBSM_B1`)
- `primary_row_count`: `480`
- `scoring_layer_count`: `40`
- `boundary_top_decile_enrichment`: `400000000.000`
- `cheap_predictor_spearman`: `1.000`
