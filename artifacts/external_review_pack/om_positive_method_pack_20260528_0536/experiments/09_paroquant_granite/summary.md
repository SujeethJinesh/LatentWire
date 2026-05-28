# ParoQuant Granite baseline

- **Hypothesis:** A rotation baseline can recover W4A16 Granite quality even when channel-set methods struggle.
- **Method implemented:** ParoQuant-style rotation plus W4A16 scoring.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Granite
- **Result/status:** PASS
- **Why it worked/failed:** High median recovery with positive CI; supports the regime-aware claim that Granite prefers rotation.
- **Supporting artifacts:** experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
