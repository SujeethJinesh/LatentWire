# KL/FFT diagnostics

- **Hypothesis:** Long-decode W4A16 failure is simple compound-error accumulation.
- **Method implemented:** Per-position KL fits and spectral/autocorrelation diagnostics.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Granite/DeepSeek/Falcon
- **Result/status:** PASS
- **Why it worked/failed:** Sublinear/sqrt-like KL and broadband FFT weaken the compounding-error and simple-predictor stories.
- **Supporting artifacts:** experimental/outlier_migrate/phase9/results/om_phase9_kl_granite_small_dense_20260519T085400Z, experimental/outlier_migrate/phase9/results/om_stage1_e1_deepseek_falcon_20260526T2335Z
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
