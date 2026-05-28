# M-PRED predictive tracker

- **Hypothesis:** One-step predictive correction beats EMA lag.
- **Method implemented:** AR/Kalman-style innovation tracker with alpha variants.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Granite/DeepSeek/Falcon
- **Result/status:** KILL
- **Why it worked/failed:** Broadband drift defeats simple prediction; predictor chased noise and underperformed EMA.
- **Supporting artifacts:** experimental/outlier_migrate/phase9/results/om_phase9_mpred_granite_reduced_20260527T201251Z, experimental/outlier_migrate/phase9/results/om_phase9_mpred_deepseek_extension_20260528T005045Z, experimental/outlier_migrate/phase9/results/om_phase9_mpred_falcon_extension_20260528T011658Z
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
