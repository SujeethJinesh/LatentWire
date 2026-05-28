# M10 position-binned scales

- **Hypothesis:** Position-binned protection/scaling can track long-decode drift with lower discontinuity than M2.
- **Method implemented:** Use position bins and precomputed scale/protection tables.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Granite
- **Result/status:** KILL
- **Why it worked/failed:** Coarse bins still induced boundary artifacts and did not beat matched controls.
- **Supporting artifacts:** experimental/outlier_migrate/phase9/results/om_phase9_m10_granite_small_vac12_20260515T085800Z
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
