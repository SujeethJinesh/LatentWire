# Channel-Set Next GPU

STATUS: BLOCKED_FOR_C_A1
GPU_FOREGROUND_AUTHORIZED: false
GPU_BACKFILL_AUTHORIZED: false for C_A1 until fresh non-confirm IDs exist

The next Channel-Set GPU action is not runnable from the current cached C_A1 gate packet. `dashboard/confirm_path_audit.md` marks the Granite tight-clip cached gate source as confirmation-contaminated.

## Required Before Launch

- Fresh non-confirm dev/gate row IDs for Granite, DeepSeek, and Falcon.
- Same-row ParoQuant baseline and tight-clip C_A1 outputs for each model.
- Native W4A16/ParoQuant provenance.
- No-gap denominator audit.
- Repro and leakage review records for backfill cache materialization, or full panel review before any claim-bearing job.

## Queue State

- `queues/gpu_foreground.yaml` must remain empty.
- `queues/gpu_backfill.yaml` keeps C_A1 first, but the C_A1 entries are blocked and `promotion_allowed=false`.
- No sidecar, killed-method, or confirmation-split job is authorized for utilization.
