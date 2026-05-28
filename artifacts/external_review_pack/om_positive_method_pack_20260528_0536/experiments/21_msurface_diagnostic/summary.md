# M-SURFACE hook diagnostic

- **Hypothesis:** An internal surface has materially lower drift than block output.
- **Method implemented:** Static hook map and cache search for internal activations.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Granite/Falcon
- **Result/status:** DEFERRED
- **Why it worked/failed:** Surfaces are hookable but cached internal activations are absent; no GPU before LAMBDA/HYST smoke.
- **Supporting artifacts:** artifacts/msurface/decision.json, artifacts/msurface/hook_map.md, artifacts/msurface/surface_drift_table.csv
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
