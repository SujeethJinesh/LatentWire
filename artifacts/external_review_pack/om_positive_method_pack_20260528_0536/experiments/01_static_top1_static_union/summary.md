# Static top-1 and static union controls

- **Hypothesis:** Static or union channel protection should recover long-decode W4A16 quality.
- **Method implemented:** Protect channels selected from fixed calibration positions or unions across calibration positions.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Granite
- **Result/status:** KILL
- **Why it worked/failed:** Static protection is not enough under drift; union/static controls either saturate budget or recover inconsistently.
- **Supporting artifacts:** experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z, experimental/outlier_migrate/phase4/results/om_phase4_20260511T054000Z
- **Caveats:** Early Phase 3/4 packets used Granite-Tiny/Small variants; use as design evidence, not final positive method.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
