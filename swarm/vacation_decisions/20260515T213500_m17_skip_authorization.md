# Vacation Decision - M17 Skip Authorization

## Situation

The human updated the Phase 9 branching logic during the active M10 run.
The new instruction says that if M10 kills and M11 also kills on
Granite-Small, M17 should be skipped and the sprint should proceed
directly to M18.

## Options Considered

- Keep the original Phase 9 tree and run M17 after an M11 kill.
- Apply the updated human instruction and skip M17 after M10 + M11 kill.

## Decision

Apply the updated branching instruction:

- If M11 passes: follow the original Branch A path.
- If M11 kills after M10 also kills: skip M17 and run M18 directly.
- If M11 kills and M18 kills: pivot to the planned negative-result
  framing.

## Rationale

M11 and M17 both primarily test whether smoother set-update dynamics can
repair the boundary-discontinuity failure pattern seen in M2 and under
test in M10. M18 tests a different mechanism, cross-tensor coupling
between activation and KV outliers, and therefore has higher remaining
information value under sprint pacing constraints.

## What Would Invalidate This

If M10 unexpectedly passes, or if M11 passes, the M17 skip condition is
not active and the original branch logic applies.
