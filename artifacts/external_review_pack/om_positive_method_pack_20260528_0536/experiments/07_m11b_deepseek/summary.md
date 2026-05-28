# M11b DeepSeek

- **Hypothesis:** M11b top-10 should generalize to dense Transformer reasoning models.
- **Method implemented:** Same M11b budget-scaling protocol on DeepSeek-R1-Distill-Qwen-1.5B.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** DeepSeek
- **Result/status:** AMBIG
- **Why it worked/failed:** Top-10 median is positive but CI crosses negative and static top-10 is competitive.
- **Supporting artifacts:** experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
