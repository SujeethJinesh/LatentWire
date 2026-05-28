# DecDEC proxy

- **Hypothesis:** DecDEC-style dynamic saliency transfers to long-reasoning W4A16.
- **Method implemented:** Algorithmic proxy for short-horizon dynamic channel identification.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Granite
- **Result/status:** KILL
- **Why it worked/failed:** Near-zero/negative recovery; short-horizon dynamic selection does not directly solve long-decode protection.
- **Supporting artifacts:** experimental/outlier_migrate/phase9/results/om_phase9_decdec_granite_small_vac12_20260517T141500Z
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
