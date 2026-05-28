# M18 activation+K coupling

- **Hypothesis:** Coupling activation channels with K-side protection captures cross-tensor sensitivity.
- **Method implemented:** Cross-tensor activation+K protected sets.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Granite
- **Result/status:** KILL
- **Why it worked/failed:** Negative recovery; extra coupling appears to inject instability rather than useful selectivity.
- **Supporting artifacts:** experimental/outlier_migrate/phase9/results/om_phase9_m18_granite_small_vac12_20260516T193500Z
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
