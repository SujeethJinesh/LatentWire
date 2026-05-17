# Phase 9 M18 Final Decision

Run directory:
`experimental/outlier_migrate/phase9/results/om_phase9_m18_granite_small_vac12_20260516T193500Z`

Decision: `KILL_M18_AMBIGUOUS`

Artifact complete: `true`

Checker:
`experimental/outlier_migrate/phase9/check_om_phase9_m18_joint_kv_activation.py`

Checker output:
`experimental/outlier_migrate/phase9/results/om_phase9_m18_granite_small_vac12_20260516T193500Z/checker_result.json`

## Primary Result

Primary regime: `m18_activation_k`

| Metric | Value |
|---|---:|
| Total trace count | `12` |
| Included positive-static-gap traces | `8` |
| No recoverable static gap | `4 / 12` |
| No-gap fraction | `0.3333333333333333` |
| Median recovery | `-0.34359084412632024` |
| Mean recovery | `-5.556008919558966` |
| CI95 | `[-14.071191710978855, 0.7406021242797479]` |

The preregistered pass threshold was median recovery `>= 0.30` with CI lower
bound `> 0.10`, while beating static activation-only, KIVI key-only, and
random coupled controls by the required margins. M18 misses the recovery bar
and has a negative median recovery.

## Controls

| Regime | Median recovery | CI95 |
|---|---:|---|
| `m18_activation_k` | `-0.34359084412632024` | `[-14.071191710978855, 0.7406021242797479]` |
| `m18_activation_kv` | `-0.29464607162152656` | `[-8.861879807739966, 0.7151491102972624]` |
| `kivi_key_only` | `-2.1741283113927237` | `[-413.9097752364582, -0.7065686932043151]` |
| `random_coupled_activation_k` | `-4.6643797310509125` | `[-330.8256289676822, -1.047125554665297]` |

M18 is less damaging than the KIVI key-only and random coupled controls by
median recovery, which is a useful diagnostic signal. It is not enough for a
positive method: the primary median remains negative and the uncertainty
interval is wide.

## Hook Coverage

The packet reports full accessible key-cache coverage for the Granite-Small
attention layers used by the M18 implementation:

- Attention layer indices: `[5, 15, 25, 35]`
- Key-cache accessible attention-layer coverage: `1.0`
- Value-cache accessible attention-layer coverage: available for the same
  layers
- Source: `past_key_values.key_cache/value_cache inspection; no model source
  modification`

## Interpretation

M18 does not establish a positive method. The result is consistent with the
cross-tensor coupling idea containing some signal, because activation+K is
less bad than K-only and random coupled controls. However, it does not recover
quality relative to the static-activation baseline under the preregistered
Granite-Small W4A16 testbed. The next step is the post-M18 analytical phase:
measure K/V migration, recovery curves, fine-bin overlap, per-layer drift, and
always-protected channel cores before authoring any new method preregistration.
