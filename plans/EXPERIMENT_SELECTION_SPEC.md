# plans/EXPERIMENT_SELECTION_SPEC.md — what to run next (expected decision value)

**Queue jobs by expected decision value, not by excitement.** The **foreground GPU queue** is scheduled by EVI (below). The `V = 2H+2N+2B+C−2S−2F−G` heuristic (`CODEX_NEXT_72H.md §13.3`) remains **only** for exploratory ordering of cached CPU screens — it never schedules GPU and never promotes.

## Required job fields
`method_id · stage · estimated_gpu_hours · estimated_cpu_hours · P_pass_subjective · claim_value_1_to_5 · novelty_after_scoop_1_to_5 · baseline_distance_1_to_5 · leakage_risk_1_to_5 · scoop_risk_1_to_5 · dependency_count · next_falsifier · figure_slot · kill_condition`.

## EVI score
```
EVI = P_pass * claim_value * novelty_after_scoop * baseline_distance
      / max(estimated_gpu_hours, 0.25)
      - leakage_risk - scoop_risk - dependency_count
```

## Scheduler rule
- **Foreground GPU** runs the highest-EVI job among `AUDITED + REVIEW-PASSED + premortem-written + planted-suite-PASS` jobs.
- **Backfill GPU** runs the highest reusable-cache value (`CODEX_NEXT_72H.md §2.5`).
- **Never** override the hard priority gates, the Phase-A baseline gate, or the do-not-queue list. **No KILLED method re-enters the GPU queue.**
- Every method agent asks: **"What is the smallest run that can change tomorrow's decision?"** (compute EVI for that minimal run, not the maximal one).

## Sequential decision (kill dead branches early; expand promising-but-underpowered)
Check after each evaluation shard:
```yaml
promote_early_if:  [lower_bound_after_min_rows > gate_margin, controls_collapse, baseline_parity_ok]
stop_for_futility_if: [upper_bound_after_current_rows < required_gate,
                       projected_mde_at_max_budget > 2 * observed_effect]
expand_if:         [point_estimate_positive, lower_bound_negative, mde_says_more_rows_can_resolve]
```
**Do not "finish the planned run"** if the confidence interval already proves the branch dead — return the GPU to the next-highest-EVI job (or backfill) immediately.

## EVI anti-gaming rules (the formula is gameable by optimistic P_pass and tiny GPU-hours)
- `P_pass_subjective` must carry a one-sentence `p_pass_basis` and be signed by the proposing agent.
- The Conductor may **shrink** `P_pass_subjective` using that agent's calibration/Brier history (§9.1).
- **Before any real Stage-1 evidence: cap `P_pass_subjective ≤ 0.25`.**
- **Before hard baselines are locked: cap `baseline_distance ≤ 2`.**
- **Before scoop/novelty review: cap `novelty_after_scoop ≤ 2`.**
- If `estimated_gpu_hours < 0.25`, also report estimated wall-clock minutes + setup/queue overhead — tiny jobs cannot hide overhead.
- EVI ranks **only among jobs that already satisfy the hard prerequisites** (AUDITED + REVIEW-PASSED + premortem + planted-PASS + Phase-A baseline). **It cannot rescue a blocked job.**

Extra required fields: `p_pass_basis · expected_decision_changed · cost_of_wrong_decision · prerequisite_status`.

## Sequential looks are for resource allocation, NOT final paper p-values
Repeated CIs after every shard are **not** automatically valid confirmatory evidence. **Stage-3 paper claims still use the pre-registered final analysis on the frozen confirmation set.** If an early promote/stop decision is used as evidence, apply an alpha-spending rule or label the look **exploratory**. Replace the brittle futility rule with:
```yaml
stop_for_futility_if:
  - conditional_power_at_max_budget < 0.20
  # equivalently: projected_mde_at_max_budget > 2 * max(abs(observed_effect), epsilon)
```
(The bare `> 2*observed_effect` form misbehaves when the observed effect is near zero or negative.)

## Positive-method bias (the queue exists to produce positive contributions)
```yaml
positive_method_bias:
  foreground_gpu_default: positive_method
  diagnostic_or_baseline_gpu_allowed_only_if:
    - it blocks a paper claim, OR
    - it is a required hard-reviewer defense (e.g. C-D1 OSC, C-E1 ParoQuant parity), OR
    - a positive method cannot be interpreted without it
  target_stage2_mix:
    positive_method_min_fraction: 0.75
    diagnostic_or_baseline_max_fraction: 0.25
```
EVI ranks within this bias: among AUDITED+REVIEW-PASSED+premortem+planted-PASS jobs, `positive_method` cards win ties and hold ≥75% of Stage-2 slots; diagnostics/baselines run only under the exceptions above. A `diagnostic_defense`/`baseline_adversary`/`ceiling_probe`/`kill_only`/`parked` card is **never** promoted to a paper claim.
