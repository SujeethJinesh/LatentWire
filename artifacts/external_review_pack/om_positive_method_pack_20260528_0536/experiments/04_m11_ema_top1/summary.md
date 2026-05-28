# M11 EMA top-1%

- **Hypothesis:** EMA smoothing removes boundary harm while staying in a 1% budget.
- **Method implemented:** EMA-smoothed top-1% protected set at scoring endpoint.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Granite
- **Result/status:** KILL
- **Why it worked/failed:** Directionally less discontinuous, but the 1% budget cannot cover the union of drifting important channels.
- **Supporting artifacts:** experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
