# Phase 6 RSPR Testbed Selection Block

**Created**: 2026-05-12
**Status**: Phase 6 RSPR skipped/blocked pending human decision

## Trigger

`swarm/goal.md` authorizes Phase 6 RSPR only if Phase 4 established a
measurable static-protection gap:

- fewer than 25% of Phase 4 traces with `no_recoverable_static_gap`
- Phase 4 not killed for measurement-design reasons

## Observed Phase 4 Facts

- Phase 4 run:
  `experimental/outlier_migrate/phase4/results/om_phase4_20260511T054000Z`
- Phase 4 decision: `KILL_OM_PHASE4_INTERVENTION_FAILS`
- Artifact complete: `true`
- Primary union median recovery: `0.0`
- Bootstrap CI95 high: `0.06954064195470389`
- `no_recoverable_static_gap_count`: `9`
- Trace count: `24`
- `no_recoverable_static_gap_fraction`: `0.375`

## Decision

Phase 6 RSPR must not run under the current authorization because the
measurable-gap condition is not met. The observed no-gap fraction `0.375`
exceeds the allowed maximum `0.25`.

This is a testbed-selection block, not a failure of RSPR data, because no
RSPR inference has been run and no Phase 6 preregistration was authored.

## Human Decision Needed

Choose one of:

1. Authorize a new measurable quantization testbed for RSPR, with a fresh
   preregistration before any data.
2. Keep RSPR out of the COLM workshop submission and frame OutlierMigrate as
   measurement/decomposition plus negative static-intervention evidence.
3. Defer RSPR to an ICLR/MLSys follow-up with a stronger quantization regime
   and explicit prior-art differentiation from HCP/CHON.

Until then, the swarm should proceed to authorized non-RSPR phases only.
