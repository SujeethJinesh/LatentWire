# ThoughtFlow-FP8 Phase 2 Retention Simulation

> Superseded artifact: this synthetic retention result is preserved for
> auditability only. It does not revive the current branch; see
> `current_decision_manifest_20260506.md` for the active stop decision.

Status: **ALIVE, but only as a synthetic Mac-local gate.** Real current-model traces are still required before a reviewer pack.

| Policy | Keep rate | Anchor recall | Phase recall | Math-state recall |
|---|---:|---:|---:|---:|
| longflow_like | 0.368 | 1.000 | 0.143 | 0.745 |
| thin_kv_like | 0.368 | 0.823 | 0.286 | 0.847 |
| rkv_like | 0.368 | 1.000 | 0.286 | 0.333 |
| thoughtflow | 0.368 | 1.000 | 1.000 | 0.329 |

## Decision

The anchor/phase policy preserves phase markers and anchors at the same keep budget where LongFlow-like, ThinKV-like, and R-KV-like proxies drop at least one protected class.
This is not accuracy evidence and not a GPU systems result. The next gate is to rerun the same policy simulator on real cached/current-model reasoning traces.
