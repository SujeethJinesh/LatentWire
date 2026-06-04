# plans/PLANTED_TESTS_SPEC.md — positive controls (the pipeline must recover a known effect)

Null-method **sentinels** (negative controls) prove the pipeline does not pass noise. **Planted-signal positive controls** prove it can *detect a real effect* — without them, a genuinely good method can be silently **killed** because the runner, aggregator, bootstrap, metric, or split is insensitive. This is probably the highest-leverage missing scientific guardrail.

**Rule:** no Stage-2 promotion is valid unless the planted-signal suite passes at the current code hash. **A failed planted test is an INFRASTRUCTURE/audit failure, not a method failure** — fix the pipeline, do not kill the method.

## Channel-Set planted tests
- `planted_tail_rescue` — artificially reduce loss on known worst traces; the aggregator **must** detect the CVaR / worst-trace improvement.
- `planted_wrong_horizon` — the correct horizon label wins; the shuffled horizon label fails.
- `planted_no_gap` — traces with zero recoverable static gap **must not** inflate the recovery ratio (denominator/CE21 audit).
- `planted_random_control` — a random policy **must not** pass the gate.

## LatentWire planted tests
- `planted_beyond_score` — inject a synthetic residual feature carrying answer info beyond `source_top1`; `delta_beyond_score` **must** become positive.
- `planted_source_copy` — a packet that encodes only `source_top1`; the `leakage_audit` **must** catch it and source-index **must** dominate.
- `planted_wrong_row` — a row-specific packet works only on matching rows; the wrong-row control **must** collapse.
- `planted_candidate_derangement` — candidate derangement **must** collapse a label-like packet.

## Gate + cadence
```yaml
gate: { no_real_method_promotes_if_planted_tests_fail: true }
```
Run the suite **at each code hash** (alongside the `§1.5` pre-launch review) **and after any change to an aggregator, metric, bootstrap, or split.** Record results in `reviews/<m>@<sha>.planted.json`; the audit agent verifies it before any Stage-2 promotion.

## Make them executable: two fixture types + concrete thresholds
```yaml
planted_suite:
  fixture_types:
    - synthetic_minimal_fixture          # fast; exercises runner+metric+bootstrap+aggregator+audit end-to-end
    - real_shape_cached_fixture_dev_only  # cached real-shaped data, dev-only, to catch shape-specific bugs
  pass_thresholds:
    channel_set:
      planted_tail_rescue:   { cvar_delta_detected_lb_gt: 0, worst_trace_delta_detected: true }
      planted_no_gap:        { recovery_ratio_unchanged_by_no_gap_rows: true }
      planted_wrong_horizon: { correct_horizon_wins: true, shuffled_horizon_fails: true }
      planted_random_control:{ promote_gate_pass_rate_eq: 0 }
    latentwire:
      planted_beyond_score:        { delta_beyond_score_lb_gt: 0 }
      planted_source_copy:         { source_index_dominates: true, packet_predicts_source_top1_acc_gt: 0.95, leakage_audit_flags: true }
      planted_wrong_row:           { wrong_row_delta_near_zero: true }
      planted_candidate_derangement:{ deranged_delta_near_zero: true }
```
**Do not plant directly into aggregate `result.json`.** Plant at the **lowest runner input layer** that still exercises the runner, metric, bootstrap, aggregator, and audit — otherwise the positive controls can "fake pass." Run both fixture types at each code hash.
