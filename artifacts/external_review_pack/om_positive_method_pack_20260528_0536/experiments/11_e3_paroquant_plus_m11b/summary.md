# E3 ParoQuant plus M11b composition

- **Hypothesis:** Rotation and dynamic channel protection compose additively.
- **Method implemented:** ParoQuant weights plus M11b top-10 protected columns and random matched controls.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Granite
- **Result/status:** KILL
- **Why it worked/failed:** Sub-additive; random matched composition was stronger, indicating overlap or interference.
- **Supporting artifacts:** experimental/outlier_migrate/phase9/results/om_stage1_e3_granite_20260526T034442Z
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
