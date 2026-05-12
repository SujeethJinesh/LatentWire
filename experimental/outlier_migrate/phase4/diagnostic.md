# OutlierMigrate Phase 4 Diagnostic

**Run**: `experimental/outlier_migrate/phase4/results/om_phase4_20260511T054000Z`
**Checker decision**: `KILL_OM_PHASE4_INTERVENTION_FAILS`
**Artifact status**: `artifact_complete=true`
**Control stop**: `false`

## Decision Summary

Phase 4 killed on the preregistered primary decision rule. The primary
migration-aware union protection regime had:

- median recovery: `0.000000000000`
- bootstrap CI95: `[0.000000000000, 0.069540641955]`
- no-recoverable-static-gap traces: `9/24` (`0.375000000000`)
- traces with recovery greater than `0.50`: `4/24`

The checker reason is:

- `median recovery 0.00000000 < 0.20`

The no-gap fraction also exceeds the preregistered `0.25` maximum, so this
packet does not support the Phase 4 positive-method claim.

## Mandatory Controls

The mandatory controls did not trigger the human-stop rule, but they also do
not rescue the union intervention:

| Regime | Median recovery | CI95 high | >0.50 traces |
| --- | ---: | ---: | ---: |
| migration-aware union | `0.000000000000` | `0.069540641955` | `4/24` |
| static-2% matched budget | `0.000000000000` | `0.509447743503` | `8/24` |
| magnitude average | `0.000000000000` | `0.047903829981` | `6/24` |

`union_outperforms_both_controls=false`, but no control exceeded union by more
than `0.10` median recovery because all medians are zero.

## Grid Sensitivity

Changing the union grid did not change the median result:

| Grid | Median recovery | CI95 |
| --- | ---: | ---: |
| sparse `{100, 5000, 10000}` | `0.000000000000` | `[-0.337218160457, 0.057480709850]` |
| primary `{100, 1000, 5000, 10000}` | `0.000000000000` | `[0.000000000000, 0.069540641955]` |
| dense `{100, 500, 1000, 2000, 5000, 7500, 10000}` | `0.000000000000` | `[-0.474740597754, 0.055115818031]` |

Dense and sparse grids both preserve the same no-gap count (`9/24`) and do
not create a robust recovery signal.

## Measurement-vs-Method Diagnosis

Phase 4 was designed to test whether the Phase 3 kill was mainly a
measurement-design artifact from Granite-Tiny being too robust under W4A16.
The Granite-Small / 512-token-window setup improved the measurement surface
only partially: `15/24` traces had a recoverable static gap, but the no-gap
fraction remained above the preregistered `25%` ceiling.

This is not clean evidence that a stronger quantization stressor would make
the same union method work. On traces with a measurable gap, the union regime
is highly unstable, including large negative recoveries. The matched-budget
and magnitude-average controls also fail to produce nonzero median recovery.
The most conservative interpretation is that simple static union protection is
not a reliable intervention under the preregistered W4A16 setting.

W3A16 or another more aggressive quantization regime might create a larger
static gap, but that would be a new preregistered intervention surface, not a
repair of Phase 4. It is not authorized by the current sprint.

## Paper Consequence

The OutlierMigrate paper must retain the characterization plus
negative-intervention framing:

- robust migration is measured on Granite and Nemotron-3;
- strict set-leaving is relevant to static protection methods;
- simple migration-aware static union protection does not recover a robust
  fraction of the W4A16 static-protection gap;
- future positive methods should target adaptive/online protection or
  refresh-style mechanisms rather than another static union-set variant.
