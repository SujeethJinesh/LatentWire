# M2 position-conditional switching

- **Hypothesis:** Hard position-conditioned protected sets can follow drift.
- **Method implemented:** Switch protected sets by decode-position bin.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Granite
- **Result/status:** KILL
- **Why it worked/failed:** Hard discontinuities lost to random controls; switching boundaries appear more harmful than stale static sets.
- **Supporting artifacts:** experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
