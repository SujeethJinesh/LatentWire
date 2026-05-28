# M-BRANCH Falcon diagnostic

- **Hypothesis:** Falcon branch-local drift is lower than post-mixer drift.
- **Method implemented:** Static Falcon branch hook map and cached evidence audit.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Falcon-H1
- **Result/status:** DEFERRED
- **Why it worked/failed:** No branch-local cache and no clear implementation reason to spend GPU before smoke.
- **Supporting artifacts:** artifacts/mbranch/decision.json, artifacts/mbranch/hook_map.md, artifacts/mbranch/branch_drift_table.csv
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
