# Channel-Set Schema Audit

AUDIT_STATUS: COMPLETE
PROMOTION_ALLOWED: false

## C_U1 Drift-As-Signal Router

Status: `MAC_FLOOR_COMPUTED_SCHEMA_BLOCKED`

The strict Mac floor materialized `500` KL trajectory rows, but the rows do not carry the paired difficulty or policy-uplift labels required by the C_U1 gate. There is no safe adapter in the repo that can infer those labels without native same-row outcome data.

Runnable replacement requirement: join KL trajectory rows to same-row policy outcomes for at least two models, then fit only on dev/gate rows with wrong-row and fixed-policy controls.

## C_W1 Fixed-Library Warmup Selector

Status: `MAC_FLOOR_COMPUTED_SCHEMA_BLOCKED`

Warmup KL rows can be parsed locally, but no cached per-policy outcome matrix exists for ParoQuant, C_A1, survival, and reject decisions on the same rows. A selector without that matrix would select against missing labels.

Runnable replacement requirement: native same-row policy matrix with ParoQuant, tight-clip C_A1, survival/static baseline, and reject outcomes.

## C_S1 Survival StableCore Denominator

Status: `PARKED_LOGGED_LOCAL_FLOOR_BLOCKED`

Only `198` eligible non-confirm per-trace rows are available locally against a `500` row floor. The floor is intentionally not lowered. A smaller design would need a new preregistered MDE and cannot be used as a powered positive-method gate.

Runnable replacement requirement: fresh identical-row denominator with static/random controls and at least `500` non-confirm rows.

## C_Y5 Defense Bundle

Status: `PARKED_LOGGED_NEEDS_NATIVE_PAIRING`

Defense inputs exist, but the three-model same-row native C_A1 versus ParoQuant packet is missing and the cached Granite member is confirm-contaminated. The defense bundle cannot validate a claim until the fresh C_A1 packet exists.
