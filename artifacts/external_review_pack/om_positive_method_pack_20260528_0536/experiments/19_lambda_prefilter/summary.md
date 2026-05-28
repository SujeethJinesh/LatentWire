# LAMBDA layerwise budget prefilter

- **Hypothesis:** Layerwise budget allocation can rescue regimes where flat M11b underperforms.
- **Method implemented:** CPU layer heterogeneity and dominant-layer allocation analysis.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** DeepSeek/Falcon
- **Result/status:** SMOKE_ONLY
- **Why it worked/failed:** Heterogeneity is sufficient to authorize smoke, but not evidence yet.
- **Supporting artifacts:** artifacts/funnel_prefilters/decision.json, artifacts/funnel_prefilters/report.md, artifacts/funnel_prefilters/smoke_traces.json
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
