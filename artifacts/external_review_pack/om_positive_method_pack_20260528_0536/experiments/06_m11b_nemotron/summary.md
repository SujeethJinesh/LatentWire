# M11b Nemotron budget-tuned EMA

- **Hypothesis:** Budget-tuned EMA transfers to Nemotron.
- **Method implemented:** Corrected W4A16 static-1% baseline plus M11b top-1/top-5/top-10.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Nemotron
- **Result/status:** PASS
- **Why it worked/failed:** Top-10 has strong positive recovery and beats static top-10, supporting a Nemotron-specific positive regime.
- **Supporting artifacts:** experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
