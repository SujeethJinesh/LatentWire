# M11b Falcon-H1

- **Hypothesis:** M11b top-10 should help Falcon-H1 parallel hybrid.
- **Method implemented:** Same M11b budget-scaling protocol on Falcon-H1.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Falcon-H1
- **Result/status:** KILL
- **Why it worked/failed:** Median recovery is near zero; Falcon appears to resist simple EMA channel protection.
- **Supporting artifacts:** experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
