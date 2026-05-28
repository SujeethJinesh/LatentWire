# E2 cross-prompt drift replication

- **Hypothesis:** Drift is AIME-specific.
- **Method implemented:** Narrowed MATH-only cross-prompt replication.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Granite/DeepSeek/Falcon
- **Result/status:** PASS
- **Why it worked/failed:** Enough replication to keep the drift story from being AIME-only, with honest narrowed scope.
- **Supporting artifacts:** experimental/outlier_migrate/phase9/results/om_stage1_e2_mathonly_20260526T1202Z
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
