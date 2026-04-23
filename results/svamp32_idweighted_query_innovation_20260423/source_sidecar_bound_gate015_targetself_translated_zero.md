# SVAMP32 Source-Innovation Sidecar Bound

- date: `2026-04-23`
- status: `oracle_sidecar_bound_fails_gate`
- target: `8/32`
- C2C teacher: `16/32`
- target_self_repair: `14/32`
- clean residual target set: `6`
- candidate: `gate015`

## Bound

- oracle target_self_repair + clean source sidecar: `15/32`
- delta vs target_self_repair: `+1`
- target losses vs target_self_repair: `0`
- clean source-necessary IDs: `1`
- failing criteria: `min_correct, min_clean_source_necessary`

## Candidate Accounting

- matched candidate correct: `10/32`
- matched C2C-only recovered: `2`
- matched clean residual recovered: `1`
- retained by source controls: `1`
- clean source-necessary IDs: `aee922049c757331`

## Interpretation

This is an oracle upper bound for a target-self-preserving sidecar. It is not a deployable method because it assumes perfect knowledge of which clean candidate wins to add.
