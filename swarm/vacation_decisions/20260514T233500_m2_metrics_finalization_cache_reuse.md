# Vacation Decision - M2 Metrics Finalization Cache Reuse

## Situation

The Phase 9 M2 Granite-4-H-Small vacation-mode run completed all expensive
scoring work, including all 12 random-bin negative-control traces, then failed
while building summary metrics because the runner used `median` without
importing it.

The completed score caches in
`experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_memfix_20260514T125800Z/score_cache/`
contain 12 prompt-indexed scores for each required regime:

- `bf16`
- `static_1pct`
- `static_3pct`
- `m2_position_conditional`
- `random_bin_assignment`

## Options Considered

1. Rerun the full M2 and random-bin dynamic scoring after importing `median`.
   This would preserve the previous code path but waste roughly 10 GPU hours
   repeating completed deterministic scoring.
2. Patch the missing import and add dynamic score-cache reuse for
   `m2_position_conditional` and `random_bin_assignment`, then rerun packet
   finalization from the completed caches.
3. Manually write the metrics packet with an ad hoc script.

## Decision

I chose option 2.

This preserves the scientific question, prompt set, model, protected sets,
thresholds, controls, and scoring results. It changes only packet
finalization mechanics after deterministic scores already exist. It is also
cleaner than an ad hoc metrics script because the normal runner and checker
remain responsible for packet structure and decision application.

## What Would Invalidate This Decision

The decision should be revisited if any score cache is incomplete, has prompt
indices other than 0-11, has a schema/regime mismatch, or if the checker
rejects the final packet. In that case, the run should be treated as an
infrastructure failure or rerun from the earliest trustworthy cached stage.
