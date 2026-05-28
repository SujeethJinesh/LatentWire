# M26 stable core

- **Hypothesis:** Channels stable across calibration positions form a reliable protected core.
- **Method implemented:** Protect stable-core channels selected by persistence across positions.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Granite
- **Result/status:** AMBIG
- **Why it worked/failed:** Positive but small/wide signal; stable channels alone do not solve drift.
- **Supporting artifacts:** experimental/outlier_migrate/phase9/results/om_phase9_m26_granite_small_vac12_20260518T203000Z
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
