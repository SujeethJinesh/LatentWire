# Phase 9 M2 Granite-Small Diagnostic

## Decision

`KILL_M2_RANDOM_CONTROL_BEATS`

The checker decision is authoritative:

`experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z/checker_result.json`

## What Ran

This packet finalizes the vacation-mode 12-trace Granite-4-H-Small M2 run.
The expensive scoring was completed in:

`experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_memfix_20260514T125800Z`

That run completed all score caches, including the random-bin negative control,
but failed during metrics finalization because the runner used `median` without
importing it. The final packet reused the completed deterministic score caches
and applied the normal checker.

Vacation decision:

`swarm/vacation_decisions/20260514T233500_m2_metrics_finalization_cache_reuse.md`

## Key Numbers

| Quantity | Value |
|---|---:|
| Artifact complete | `true` |
| Vacation trace count | `12` |
| Positive static-gap traces included | `8` |
| No-recoverable-static-gap traces | `4 / 12` |
| No-recoverable-static-gap fraction | `0.3333333333333333` |
| M2 median recovery | `-0.8668373133910525` |
| M2 bootstrap CI95 | `[-3.4352892513350524, 0.5952386657662287]` |
| Static-3% matched-cost median recovery | `-0.6259276389593869` |
| Random-bin median recovery | `-0.19928902322343722` |
| M2 minus static-3% median | `-0.24090967443166567` |
| M2 minus random-bin median | `-0.6675482901676153` |

## Interpretation

M2 does not provide a positive method on Granite-4-H-Small. The random-bin
negative control is also not good in absolute terms, but it is substantially
less bad than M2 and beats M2 by more than the preregistered `0.10` median
recovery threshold. This triggers the explicit random-control kill.

This result weakens the position-conditioned set-switching intervention, not
the measurement premise. Phase 9 Step 9.0 still supports the underlying
decode-position channel drift story; M2 shows that a simple fixed
position-bin switching rule is not enough to exploit it under this W4A16
testbed.

## Next Step Under Vacation Mode

Per vacation-mode D3, do not run M2 on additional models after this
Granite-Small kill. Start paper Draft 0 first, then attempt M10 on
Granite-Small as the next method branch.
