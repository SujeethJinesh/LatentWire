# WJAC offline prefilter

- **Hypothesis:** Weight-column norm sensitivity changes top-k enough to justify GPU scoring.
- **Method implemented:** Score q_i * EMA(x_i^2) with q_i=||W[:,i]||^2; compare overlap/rank/churn offline.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** all cached slices
- **Result/status:** KILL
- **Why it worked/failed:** Corrected two-of-four kill diagnostics triggered on every covered model/slice.
- **Supporting artifacts:** artifacts/wjac_prefilter/decision.json, artifacts/wjac_prefilter/report.md, artifacts/wjac_prefilter/wjac_scores.parquet
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
