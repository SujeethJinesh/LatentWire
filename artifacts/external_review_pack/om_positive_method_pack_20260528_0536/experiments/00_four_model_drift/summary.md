# Four-model strict channel-set drift

- **Hypothesis:** Static top activation-channel sets persist through long decode.
- **Method implemented:** Measure strict set-leaving / migration between early and late decode channel sets.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Granite/Nemotron/DeepSeek/Falcon
- **Result/status:** PASS
- **Why it worked/failed:** The diagnostic worked: all measured model families show large channel-set movement, which motivates dynamic or regime-aware protection.
- **Supporting artifacts:** experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z, experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z, experimental/outlier_migrate/phase7/results/om_phase7_falcon_h1_20260512T223600Z, experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z
- **Caveats:** The earliest packets report migration fraction; later paper framing reports strict set-leaving with revised normalizations.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
