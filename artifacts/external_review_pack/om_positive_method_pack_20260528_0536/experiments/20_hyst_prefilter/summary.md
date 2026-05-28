# HYST churn prefilter

- **Hypothesis:** Hysteresis can reduce harmful protected-set churn while preserving local pool stability.
- **Method implemented:** CPU churn and local-pool stability analysis.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** DeepSeek/Falcon
- **Result/status:** SMOKE_ONLY
- **Why it worked/failed:** Churn/local-stability pattern authorizes smoke; no endpoint scoring yet.
- **Supporting artifacts:** artifacts/funnel_prefilters/decision.json, artifacts/funnel_prefilters/report.md, artifacts/funnel_prefilters/smoke_traces.json
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
