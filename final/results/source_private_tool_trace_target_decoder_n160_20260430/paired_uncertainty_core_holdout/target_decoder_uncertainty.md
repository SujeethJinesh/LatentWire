# Source-Private Target-Decoder Paired Uncertainty

- pass gate: `True`
- rows: `2`
- pass rows: `2`
- min matched-target: `0.444`
- min matched-best-control: `0.444`
- min CI95 low vs target: `0.369`
- min CI95 low vs best control: `0.369`
- min valid prediction rate: `1.000`
- max p50 latency ms: `2670.306`

## Rows

| Surface | N | Matched | Target | Best control | Matched-target | Matched-control | CI low target | CI low control | Valid | p50 ms | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core_seed29_qwen3_n160_all_controls_cpu | 160 | 0.694 | 0.250 | 0.250 | 0.444 | 0.444 | 0.369 | 0.369 | 1.000 | 2670.3 | `True` |
| holdout_seed30_qwen3_n160_all_controls_cpu | 160 | 0.719 | 0.250 | 0.263 | 0.469 | 0.456 | 0.394 | 0.381 | 1.000 | 2451.1 | `True` |

## Interpretation

This gate asks whether a frozen target LLM can consume the compact source packet beyond target priors and same-byte/source-destroyed controls. It is a reviewer-defense receiver gate, not a systems-speed claim.

## Pass Rule

Every target-decoder run must pass its point gate, preserve exact-ID parity, have matched valid prediction rate >=0.95, and have paired CI95 lower bounds above +0.10 versus both target-only and the strongest source-destroying or matched-byte text control.
