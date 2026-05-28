# Falcon Channel Fallbacks

Created: `2026-05-28T15:37:22Z`

## Status

- Paper readiness: not ICLR-ready.
- Current story: V1 made rotation the priority; Falcon channel fallbacks are lower priority until ParoQuant Falcon smoke completes.
- Blocking gap: Falcon has no positive method. Existing M11b top-10 recovery is weak (`0.0439`) and all current channel policies are near zero.

No GPU jobs were run. This packet prepares LAMBDA and HYST configs for a gated fallback.

## Decision

**Decision: `HOLD_UNTIL_PAROQUANT_FALCON_SMOKE`.**

- LAMBDA: configured but deferred. It preserves total protected-channel budget but lacks cached causal per-layer recovery headroom.
- HYST: configured and CPU-promoted if a channel fallback is needed. It targets Falcon protected-set churn with exit margin `m=5`.

If ParoQuant Falcon passes smoke, keep these fallbacks dormant. If ParoQuant Falcon is weak, run HYST first; run LAMBDA only if orchestrator wants the full smoke packet or a Falcon late-layer budget prior.

## Budget Discipline

LAMBDA preserves the M11b top-10 total budget:

- layers: `36`
- flat budget per layer: `103`
- total protected channels: `3708`
- LAMBDA budget range: `68-160`

HYST preserves the per-layer active budget:

- active protected channels per layer: `103`
- enter threshold: top `10%`
- selected exit margin: `5` percentage points
- exit threshold: top `15%`, rank count `154`
- predicted late churn reduction: `0.796`
- expected active `K_eff`: `103` per layer; stale pool may reach rank `154` but active protected count is capped.

## Recommended Order If Falcon Needs Channel Rescue

1. HYST smoke with margin `5`.
2. LAMBDA smoke only if HYST fails/ambiguous and ParoQuant Falcon is weak, or if the orchestrator explicitly wants a layer-budget comparison.
3. LAMBDA+HYST only if both single factors pass individually.

