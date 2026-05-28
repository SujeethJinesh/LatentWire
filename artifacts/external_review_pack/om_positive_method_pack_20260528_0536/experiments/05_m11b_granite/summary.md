# M11b Granite budget-tuned EMA

- **Hypothesis:** Increasing EMA budget to 5-10% rescues Granite.
- **Method implemented:** EMA-smoothed protection at top-1/top-5/top-10 budgets.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Granite
- **Result/status:** AMBIG
- **Why it worked/failed:** Top-5 median is positive but CI is wide; no-gap traces and trace heterogeneity limit confidence.
- **Supporting artifacts:** experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
